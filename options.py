import os
import argparse

import torch


class BaseOptions():
	def __init__(self):
		self.initialized = False

	def initialize(self, parser):
		parser.add_argument('-n', '--name', type=str, required=True, help='tracker name')
		# GPU
		parser.add_argument('--cuda', action='store_true', default=False, help='enable GPU')
		self.initialized = True
		return parser

	def gather_options(self):
		if not self.initialized:
			parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
			parser = self.initialize(parser)

		self.parser = parser
		return parser.parse_args()

	def print_options(self, opt):
		message = ''
		message += '---------------------------- Options --------------------------\n'
		for k, v in sorted(vars(opt).items()):
			comment = ''
			default = self.parser.get_default(k)
			if v != default:
				comment = '\t[default: {}]'.format(str(default))
			message += '{:>15}: {:<25}{}\n'.format(str(k), str(v), comment)
		message += '---------------------------- End ------------------------------'
		print(message)
		return message

	def save_options(self, opt, message):
		os.makedirs(os.path.join(opt.log_dir), exist_ok=True)
		with open(os.path.join(opt.log_dir, 'options.txt'), 'wt') as f:
			command = ''
			for k, v in sorted(vars(opt).items()):
				command += '--{} {} '.format(k, str(v))
			command += '\n'
			f.write(command)
			f.write(message)
			f.write('\n')

	def parse(self):
		opt = self.gather_options()

		if opt.cuda and torch.cuda.is_available():
			opt.device = 'cuda'
		else:
			opt.cuda = False
			opt.device = 'cpu'

		self.opt = opt
		return self.opt


class TrainOptions(BaseOptions):
	def initialize(self, parser):
		parser = BaseOptions.initialize(self, parser)
		# data
		parser.add_argument('-d', '--dataset', type=str, default='got10k', help='dataset: ilsvrc | got10k')
		parser.add_argument('-j', '--num_workers', type=int, default=4, help='number of workers to laod data')
		# model
		parser.add_argument('-w', '--weight', type=str,  default=None, help='pre-trained model weight path')
		# hyperparameter
		parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs')
		parser.add_argument('--batch_size', type=int, default=8, help='batch size')
		# log
		parser.add_argument('-l', '--log_dir', type=str, default='logs', help='log directory')
		parser.add_argument('--print_freq', type=int, default=100, help='frequency to print')
		return parser

	def parse(self):
		opt = BaseOptions.parse(self)

		self.opt = opt
		message = self.print_options(opt)
		self.save_options(opt, message)
		return self.opt


class TestOptions(BaseOptions):
	def initialize(self, parser):
		parser = BaseOptions.initialize(self, parser)
		# experiments
		parser.add_argument('-e', '--exps', type=str, nargs='*', help='experiments: otb2013 | otb2015 | vot2018 | got10k')
		# model
		parser.add_argument('-w', '--weight', type=str, required=True, help='model weight path')
		# log
		parser.add_argument('--visualize', action='store_true', default=False, help='enable visualization')
		return parser

	def parse(self):
		opt = BaseOptions.parse(self)

		self.opt = opt
		self.print_options(opt)
		return self.opt
	