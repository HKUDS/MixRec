import argparse

def parse_args():
	parser = argparse.ArgumentParser(description='Model Params')
	parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
	parser.add_argument('--batch', default=512, type=int, help='batch size')
	parser.add_argument('--reg', default=1e-4, type=float, help='weight decay regularizer')
	parser.add_argument('--epoch', default=100, type=int, help='number of epochs')
	parser.add_argument('--decay', default=0.96, type=float, help='weight decay rate')
	parser.add_argument('--save_path', default='tem', help='file name to save model and training record')
	parser.add_argument('--latdim', default=16, type=int, help='embedding size')

	parser.add_argument('--sampNum', default=40, type=int, help='batch size for sampling')
	parser.add_argument('--gnn_layer', default=2, type=int, help='number of gnn layers')
	parser.add_argument('--hyperNum', default=128, type=int, help='number of hyper edges')
	parser.add_argument('--trnNum', default=10000, type=int, help='number of training instances per epoch')
	parser.add_argument('--load_model', default=None, help='model name to load')
	parser.add_argument('--shoot', default=10, type=int, help='K of top k')
	parser.add_argument('--data', default='beibei', type=str, help='name of dataset')
	parser.add_argument('--deep_layer', default=2, type=int, help='number of deep layers to make the final prediction')
	parser.add_argument('--mult', default=1, type=float, help='multiplier for the result')
	parser.add_argument('--keepRate', default=0.5, type=float, help='rate for dropout')
	parser.add_argument('--graphSampleN', default=20000, type=int, help='use 20000 for training and 40000 for testing, empirically')
	parser.add_argument('--testgraphSampleN', default=40000, type=int, help='use 20000 for training and 40000 for testing, empirically')

	parser.add_argument('--tstEpoch', default=3, type=int, help='number of epoch to test while training')
	parser.add_argument('--leaky', default=0.5, type=float, help='slope for leaky relu')
	parser.add_argument('--temp', default=1, type=float, help='temperature in node ssl loss')
	parser.add_argument('--tempGlobal', default=1, type=float, help='temperature in global ssl loss')

	parser.add_argument('--ssl_reg', default=1e-5, type=float, help='reg weight for ssl loss')
	parser.add_argument('--sslGlobal_reg', default=1e-5, type=float, help='reg weight for ssl loss')
	
	parser.add_argument('--tstNum', default=-1, type=int, help='Numer of negative samples while testing, -1 for all negatives')
	
	return parser.parse_args()
args = parse_args()
# ijcai
# args.user = 423423
# args.item = 874328
# beibei
# args.user = 21716
# args.item = 7977
# Tmall
# args.user = 114503
# args.item = 66706

args.decay_step = args.trnNum//args.batch
