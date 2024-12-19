import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from Params import args
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
import Utils.NNLayers as NNs
from Utils.NNLayers import FC, Regularize, Activate, Dropout, Bias, getParam, defineParam, defineRandomNameParam
from MixRec.DataHandler import negSamp, transpose, DataHandler, transToLsts
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
import pickle

# np.random.seed(2021)
class Recommender:
	def __init__(self, sess, handler):
		self.sess = sess
		self.handler = handler

		print('USER', args.user, 'ITEM', args.item)
		self.metrics = dict()
		mets = ['Loss', 'preLoss', 'HR', 'NDCG']
		for met in mets:
			self.metrics['Train' + met] = list()
			self.metrics['Test' + met] = list()

	def makePrint(self, name, ep, reses, save):
		ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
		for metric in reses:
			val = reses[metric]
			ret += '%s = %.4f, ' % (metric, val)
			tem = name + metric
			if save and tem in self.metrics:
				self.metrics[tem].append(val)
		ret = ret[:-2] + '  '
		return ret

	def run(self):
		self.prepareModel()
		log('Model Prepared')
		if args.load_model != None:
			self.loadModel()
			stloc = len(self.metrics['TrainLoss']) * args.tstEpoch - (args.tstEpoch - 1)
		else:
			stloc = 0
			init = tf.global_variables_initializer()
			self.sess.run(init)
			log('Variables Inited')
		for ep in range(stloc, args.epoch):
			test = (ep % args.tstEpoch == 0)
			reses = self.trainEpoch()
			log(self.makePrint('Train', ep, reses, test))
			if test:
				reses = self.testEpoch()
				log(self.makePrint('Test', ep, reses, test))
			if ep % args.tstEpoch == 0:
				self.saveHistory()
			print()
		reses = self.testEpoch()
		log(self.makePrint('Test', args.epoch, reses, True))
		self.saveHistory()

	def messagePropagate(self, lats, adjs):
		newLats = []
		for b in range(args.behNum):
			lat = tf.sparse.sparse_dense_matmul(adjs[b], lats)
			# lat = FC(lat, args.latdim, reg=True, useBias=True, activation=self.actFunc)
			newLats.append(lat)
		return newLats

	def messagePropagate(self, lats, adjs):
		newLats = []
		weights = [0.05, 0.8, 0.15]
		# weights = [10, 3, 1]
		# weights = [30, 10, 3]
		# weights = [3, 1, 0.3]
		for b in range(args.behNum):
			lat = tf.sparse.sparse_dense_matmul(adjs[b], lats) * weights[b]
			# lat = FC(lat, args.latdim, reg=True, useBias=True, activation=self.actFunc)
			newLats.append(lat)
		return newLats
		
	def hyperPropagate(self, lats, adj):
		hyperEdges = []
		hyperNodes = []
		for b in range(args.behNum):
			hyperEdge = Activate(tf.transpose(adj) @ lats[b], self.actFunc)
			# hyperEdge = tf.transpose(FC(tf.transpose(hyperEdge1), args.hyperNum, activation=self.actFunc))
			hyperEdges.append(hyperEdge)
			hyperNodes.append(adj @ hyperEdge)
		
		return hyperEdges, hyperNodes

	def edgeDropout(self, mats):
		def dropOneMat(mat):
			indices = mat.indices
			values = mat.values
			shape = mat.dense_shape
			newVals = tf.nn.dropout(values, self.keepRate)
			return tf.sparse.SparseTensor(indices, newVals, shape)
		dropMats = []
		for b in range(args.behNum):
			dropMats.append(dropOneMat(mats[b]))
		return dropMats

	def ours(self):
		uEmbed0 = NNs.defineParam('uEmbed0', [args.user, args.latdim], reg=True)
		iEmbed0 = NNs.defineParam('iEmbed0', [args.item, args.latdim], reg=True)
		uhyper = NNs.defineParam('uhyper', [args.latdim, args.hyperNum], reg=True)
		ihyper = NNs.defineParam('ihyper', [args.latdim, args.hyperNum], reg=True)
		uuHyper = (uEmbed0 @ uhyper)
		iiHyper = (iEmbed0 @ ihyper)
		ulats = [uEmbed0]
		ilats = [iEmbed0]
		dis_ulats = [uEmbed0]
		dis_ilats = [iEmbed0]

		def calcNodeSSL(nodelat, _nodelat, nodelat_):
			posScore = tf.exp(tf.reduce_sum(nodelat * _nodelat, axis=1) / args.temp)
			negScore = tf.reduce_sum(tf.exp(nodelat @ tf.transpose(nodelat_) / args.temp), axis=1)
			uLoss = tf.reduce_sum(-tf.log(posScore / (negScore + 1e-8) + 1e-8))
			return uLoss

		def calcGlobalSSL(targetHyperEU, posHyperEU, negHyperEU):
			posScore = tf.exp(tf.reduce_sum(targetHyperEU * posHyperEU) / args.tempGlobal)
			negScore = tf.exp(tf.reduce_sum(targetHyperEU * negHyperEU) / args.tempGlobal)
			uLoss = tf.reduce_sum(-tf.log(posScore / (posScore + negScore + 1e-8) + 1e-8))
			return uLoss

		sslloss = 0
		sslloss_global = 0
		uniqUids, _ = tf.unique(self.uids)
		uniqIids, _ = tf.unique(self.iids)
		
		dis_uhyper = tf.gather(uhyper, tf.random.shuffle(tf.range(tf.shape(uhyper)[0])))
		dis_ihyper = tf.gather(ihyper, tf.random.shuffle(tf.range(tf.shape(ihyper)[0])))
		dis_uEmbd = tf.gather(uEmbed0, tf.random.shuffle(tf.range(tf.shape(uEmbed0)[0])))
		dis_iEmbd = tf.gather(iEmbed0, tf.random.shuffle(tf.range(tf.shape(iEmbed0)[0])))
		dis_uuHyper = dis_uEmbd @ dis_uhyper
		dis_iiHyper = dis_iEmbd @ dis_ihyper

		def dropOneMat(mat):
			indices = mat.indices
			values = mat.values
			shape = mat.dense_shape
			newVals = tf.nn.dropout(values, self.keepRate)
			return tf.sparse.SparseTensor(indices, newVals, shape)

		def onehyperPropagate(lat, adj):
			hyperEdge1 = Activate(tf.transpose(adj) @ lat, self.actFunc)
			hyperEdge = tf.transpose(FC(tf.transpose(hyperEdge1), args.hyperNum, activation=self.actFunc))
			hyperNode = adj @ hyperEdge
			return hyperEdge, hyperNode

		def onemessagePropagate(lat, adj):
			lat = tf.sparse.sparse_dense_matmul(adj, lat)
			return lat

		for i in range(args.gnn_layer):
			ulat_behs = self.messagePropagate(ilats[-1], self.edgeDropout(self.adjs))
			ilat_behs = self.messagePropagate(ulats[-1], self.edgeDropout(self.tpAdjs))
			ulat = tf.add_n(ulat_behs)
			ilat = tf.add_n(ilat_behs)

			hyperUEdge_behs, hyperULat_behs = self.hyperPropagate(ulat_behs, uuHyper)
			hyperIEdge_behs, hyperILat_behs = self.hyperPropagate(ilat_behs, iiHyper)
			hyperULat = tf.add_n(hyperULat_behs)
			hyperILat = tf.add_n(hyperILat_behs)
			
			ulats.append(ulat + hyperULat + ulats[-1])
			ilats.append(ilat + hyperILat + ilats[-1])

			targetHyperEU = tf.nn.l2_normalize(hyperUEdge_behs[args.behNum - 1])
			targetHyperEI = tf.nn.l2_normalize(hyperIEdge_behs[args.behNum - 1])
			targetHyperEU = tf.reduce_sum(targetHyperEU, axis=0) 
			targetHyperEI = tf.reduce_sum(targetHyperEI, axis=0) 

			disNodeU = onemessagePropagate(dis_ilats[-1], dropOneMat(self.disAdj))
			disNodeI = onemessagePropagate(dis_ulats[-1], dropOneMat(self.disTpAdj))
			negHyperEU, negNodeU = onehyperPropagate(disNodeU, dis_uuHyper)
			negHyperEI, negNodeI = onehyperPropagate(disNodeI, dis_iiHyper)
			
			dis_ulats.append(disNodeU + negNodeU + dis_ulats[-1])
			dis_ilats.append(disNodeI + negNodeI + dis_ilats[-1])

			negHyperEU = tf.nn.l2_normalize(negHyperEU)
			negHyperEI = tf.nn.l2_normalize(negHyperEI)
			negHyperEU = tf.reduce_sum(negHyperEU, axis=0) 
			negHyperEI = tf.reduce_sum(negHyperEI, axis=0) 

			targetNodelatU = hyperULat_behs[-1]
			targetNodelatI = hyperILat_behs[-1]
			# batch
			nodeULat = tf.nn.l2_normalize(tf.nn.embedding_lookup(targetNodelatU, uniqUids), axis=1)
			nodeILat = tf.nn.l2_normalize(tf.nn.embedding_lookup(targetNodelatI, uniqIids), axis=1)
			# all
			nodeULat_ = tf.nn.l2_normalize(targetNodelatU, axis=1)
			nodeILat_ = tf.nn.l2_normalize(targetNodelatI, axis=1)
			for b in range(args.behNum - 1):
				posHyperEU = tf.nn.l2_normalize(hyperUEdge_behs[b])
				posHyperEI = tf.nn.l2_normalize(hyperIEdge_behs[b])
				posHyperEU = tf.reduce_sum(posHyperEU, axis=0) 
				posHyperEI = tf.reduce_sum(posHyperEI, axis=0) 
				uLoss_global = calcGlobalSSL(targetHyperEU, posHyperEU, negHyperEU)
				iLoss_global = calcGlobalSSL(targetHyperEI, posHyperEI, negHyperEI)
				sslloss_global += uLoss_global + iLoss_global
				
				_nodeULat = tf.nn.l2_normalize(tf.nn.embedding_lookup(hyperULat_behs[b], uniqUids), axis=1)
				_nodeILat = tf.nn.l2_normalize(tf.nn.embedding_lookup(hyperILat_behs[b], uniqIids), axis=1)

				uweight = tf.reshape(FC(_nodeULat, args.latdim * args.latdim, reg=True, activation=self.actFunc), [-1, args.latdim, args.latdim])
				iweight = tf.reshape(FC(_nodeILat, args.latdim * args.latdim, reg=True, activation=self.actFunc), [-1, args.latdim, args.latdim])
				_nodeULat = tf.reduce_sum(tf.multiply(tf.expand_dims(_nodeULat, axis=-1), uweight), axis=1)
				_nodeILat = tf.reduce_sum(tf.multiply(tf.expand_dims(_nodeILat, axis=-1), iweight), axis=1)	
				# _nodeULat = FC(_nodeULat, args.latdim, reg=True, useBias=True, activation=self.actFunc)
				# _nodeILat = FC(_nodeILat, args.latdim, reg=True, useBias=True, activation=self.actFunc)

				uLoss = calcNodeSSL(nodeULat, _nodeULat, nodeULat)
				iLoss = calcNodeSSL(nodeILat, _nodeILat, nodeILat)
				sslloss += uLoss + iLoss
			sslloss_global = sslloss_global / (args.behNum - 1)
			sslloss = sslloss / (args.behNum - 1)
		
		ulat = tf.add_n(ulats)
		ilat = tf.add_n(ilats)

		pckUlat = tf.nn.embedding_lookup(ulat, self.uids)
		pckIlat = tf.nn.embedding_lookup(ilat, self.iids)
		preds = tf.reduce_sum(pckUlat * pckIlat, axis=-1)

		return preds, sslloss, sslloss_global

	def prepareModel(self):
		self.keepRate = tf.placeholder(dtype=tf.float32, shape=[])
		NNs.leaky = args.leaky
		self.actFunc = 'twoWayLeakyRelu6'
		adjs = self.handler.trnMats

		idx, data, shape = transToLsts(self.handler.randomMat)
		self.disAdj = tf.sparse.SparseTensor(idx, data, shape)
		idx, data, shape = transToLsts(transpose(self.handler.randomMat))
		self.disTpAdj = tf.sparse.SparseTensor(idx, data, shape)

		self.adjs = []
		self.tpAdjs = []

		for b in range(args.behNum):
			idx, data, shape = transToLsts(adjs[b])
			self.adjs.append(tf.sparse.SparseTensor(idx, data, shape))
			idx, data, shape = transToLsts(transpose(adjs[b]))
			self.tpAdjs.append(tf.sparse.SparseTensor(idx, data, shape))

		self.uids = tf.placeholder(name='uids', dtype=tf.int32, shape=[None])
		self.iids = tf.placeholder(name='iids', dtype=tf.int32, shape=[None])
		
		self.preds, sslloss, sslloss_global = self.ours()
		sampNum = tf.shape(self.uids)[0] // 2
		posPred = tf.slice(self.preds, [0], [sampNum])
		negPred = tf.slice(self.preds, [sampNum], [-1])
		self.preLoss = tf.reduce_sum(tf.maximum(0.0, 1.0 - (posPred - negPred))) / args.batch
		self.regLoss = args.reg * Regularize() + args.ssl_reg * sslloss + args.sslGlobal_reg * sslloss_global
		self.loss = self.preLoss + self.regLoss

		globalStep = tf.Variable(0, trainable=False)
		learningRate = tf.train.exponential_decay(args.lr, globalStep, args.decay_step, args.decay, staircase=True)
		self.optimizer = tf.train.AdamOptimizer(learningRate).minimize(self.loss, global_step=globalStep)

	def sampleTrainBatch(self, batchIds, itmnum, label):
		preSamp = list(np.random.permutation(itmnum))
		temLabel = label[batchIds].toarray()
		batch = len(batchIds)
		temlen = batch * 2 * args.sampNum
		uIntLoc = [None] * temlen
		iIntLoc = [None] * temlen
		cur = 0
		for i in range(batch):
			posset = np.reshape(np.argwhere(temLabel[i]!=0), [-1])
			negset = negSamp(temLabel[i], preSamp)
			poslocs = np.random.choice(posset, args.sampNum)
			neglocs = np.random.choice(negset, args.sampNum)
			for j in range(args.sampNum):
				uIntLoc[cur] = uIntLoc[cur+temlen//2] = batchIds[i]
				iIntLoc[cur] = poslocs[j]
				iIntLoc[cur+temlen//2] = neglocs[j]
				cur += 1
		return uIntLoc, iIntLoc

	def trainEpoch(self):
		num = args.user
		sfIds = np.random.permutation(self.handler.trnUsrs)[:args.trnNum]
		epochLoss, epochPreLoss = [0] * 2
		num = len(sfIds)
		steps = int(np.ceil(num / args.batch))
		feed_dict = {}
		for i in range(steps):
			st = i * args.batch
			ed = min((i+1) * args.batch, num)
			batIds = sfIds[st: ed]

			target = [self.optimizer, self.preLoss, self.regLoss, self.loss]
			uLocs, iLocs = self.sampleTrainBatch(batIds, self.handler.trnMats[-1].shape[1], self.handler.trnMats[-1])
			feed_dict[self.uids] = uLocs
			feed_dict[self.iids] = iLocs
			feed_dict[self.keepRate] = args.keepRate

			res = self.sess.run(target, feed_dict=feed_dict, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))

			preLoss, regLoss, loss = res[1:]

			epochLoss += loss
			epochPreLoss += preLoss
			log('Step %d/%d: loss = %.2f, regLoss = %.2f         ' % (i, steps, loss, regLoss), save=False, oneline=True)
		ret = dict()
		ret['Loss'] = epochLoss / steps
		ret['preLoss'] = epochPreLoss / steps
		return ret

	def sampleTestBatch(self, batchIds, label, tstInt):
		batch = len(batchIds)
		temTst = tstInt[batchIds]
		temLabel = label[batchIds].toarray()
		temlen = batch * 100
		uIntLoc = [None] * temlen
		iIntLoc = [None] * temlen
		tstLocs = [None] * batch
		cur = 0
		for i in range(batch):
			posloc = temTst[i]
			negset = np.reshape(np.argwhere(temLabel[i]==0), [-1])
			rdnNegSet = np.random.permutation(negset)[:99]
			locset = np.concatenate((rdnNegSet, np.array([posloc])))
			tstLocs[i] = locset
			for j in range(100):
				uIntLoc[cur] = batchIds[i]
				iIntLoc[cur] = locset[j]
				cur += 1
		return uIntLoc, iIntLoc, temTst, tstLocs

	def testEpoch(self):
		epochHit, epochNdcg = [0] * 2
		ids = self.handler.tstUsrs
		num = len(ids)
		tstBat = args.batch
		steps = int(np.ceil(num / tstBat))
		feed_dict = {}
		for i in range(steps):
			st = i * tstBat
			ed = min((i+1) * tstBat, num)
			batIds = ids[st: ed]
			uLocs, iLocs, temTst, tstLocs = self.sampleTestBatch(batIds, self.handler.label, self.handler.tstInt)
			feed_dict[self.uids] = uLocs
			feed_dict[self.iids] = iLocs
			feed_dict[self.keepRate] = 1.0
			preds = self.sess.run(self.preds, feed_dict=feed_dict, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
			hit, ndcg = self.calcRes(np.reshape(preds, [ed-st, 100]), temTst, tstLocs)
			epochHit += hit
			epochNdcg += ndcg
			log('Steps %d/%d: hit = %d, ndcg = %d          ' % (i, steps, hit, ndcg), save=False, oneline=True)
		ret = dict()
		ret['HR'] = epochHit / num
		ret['NDCG'] = epochNdcg / num
		return ret

	def calcRes(self, preds, temTst, tstLocs):
		hit = 0
		ndcg = 0
		for j in range(preds.shape[0]):
			predvals = list(zip(preds[j], tstLocs[j]))
			predvals.sort(key=lambda x: x[0], reverse=True)
			shoot = list(map(lambda x: x[1], predvals[:args.shoot]))
			if temTst[j] in shoot:
				hit += 1
				ndcg += np.reciprocal(np.log2(shoot.index(temTst[j])+2))
		return hit, ndcg
	
	def saveHistory(self):
		if args.epoch == 0:
			return
		with open('History/' + args.save_path + '.his', 'wb') as fs:
			pickle.dump(self.metrics, fs)

		saver = tf.train.Saver()
		saver.save(self.sess, 'Models/' + args.save_path)
		log('Model Saved: %s' % args.save_path)

	def loadModel(self):
		saver = tf.train.Saver()
		saver.restore(sess, 'Models/' + args.load_model)
		with open('History/' + args.load_model + '.his', 'rb') as fs:
			self.metrics = pickle.load(fs)
		log('Model Loaded')	

if __name__ == '__main__':
	logger.saveDefault = True
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	log('Start')
	handler = DataHandler()
	handler.LoadData()
	log('Load Data')

	with tf.Session(config=config) as sess:
		tf.set_random_seed(2021)
		recom = Recommender(sess, handler)
		recom.run()
