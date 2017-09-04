from	traceback		import print_exc
import	os
import	shutil
import	errno
from	multiprocessing	import Pool
import	string
import	random
import	numpy	as	np
import	sys
sys.path.append("game/")

import skimage
from skimage import transform, color, exposure
from skimage.color import rgb2hsv

import keras
from keras			import	regularizers
from keras.models	import	Sequential, Model
from keras.layers	import	Dense, Flatten, Activation, Input
from keras.layers	import	Conv2D, Conv2DTranspose,	UpSampling2D , MaxPooling2D	,	MaxPooling1D	,	AveragePooling2D	,	MaxoutDense	,	Average	,	Reshape	,	GlobalAveragePooling1D	,	Activation	,	Add	,	Lambda	,	BatchNormalization	,	Conv3D
from keras.optimizers import RMSprop	,	Nadam	,	SGD
import	keras.backend	as K
# from	keras.backend	import	pool2d
from keras.callbacks import LearningRateScheduler, History
import tensorflow as tf

import pygame
import wrapped_flappy_bird as game

import	threading
from	threading import Thread

import time
import math
from keras.callbacks import TensorBoard

import pydot
from keras.utils import plot_model

def	rnd_String():
	return	''.join	(
					random.choice	(
										string.ascii_uppercase
									+	string.digits
									)	for _ in range(5)
					)

shutil.copy( 'flap.py' , 'history/' + rnd_String() + '.py')

class color:
	'''color class:
	reset all color with color.reset
	two subclasses fg for foreground and bg for background.
	use as color.subclass.colorname.
	i.e. color.fg.red or color.bg.green
	also, the generic bold, disable, underline, reverse, strikethrough,
	and invisible work with the main class
	i.e. color.bold
	'''
	reset='\033[0m'
	bold='\033[01m'
	disable='\033[02m'
	underline='\033[04m'
	reverse='\033[07m'
	strikethrough='\033[09m'
	invisible='\033[08m'

class bg:
	black='\033[40m'
	red='\033[41m'
	green='\033[42m'
	orange='\033[43m'
	blue='\033[44m'
	purple='\033[45m'
	cyan='\033[46m'
	lightgrey='\033[47m'

class fg:
	black='\033[30m'
	red='\033[31m'
	green='\033[32m'
	orange='\033[33m'
	blue='\033[34m'
	purple='\033[35m'
	cyan='\033[36m'
	lightgrey='\033[37m'
	darkgrey='\033[90m'
	lightred='\033[91m'
	lightgreen='\033[92m'
	yellow='\033[93m'
	lightblue='\033[94m'
	pink='\033[95m'
	lightcyan='\033[96m'


# from resnet import ResnetBuilder

rnd_name	=	''.join	(
						random.choice	(
											string.ascii_uppercase
										+	string.digits
										)	for _ in range(4)
						)

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction =	0.9
config.gpu_options.per_process_gpu_memory_fraction =	0.6
set_session( tf.Session(config=config) )



game_state	=	game.GameState()
frame,__,__	=	game_state.frame_step( [1,0] )
# frame		=	frame[ 58: -70 + 16 ]
# frame		=	frame[ 58: -70 ]	#	some parts of our view are irrelevant
# frame		=	frame[ : , 1 ]	-	frame[ : , 0 ]	#	combine into a single channel without loosing clarity on colors
# frame		=	np.delete(frame	,	[1,2]	,	axis=2)		#	we don't need blue
# frame		=	frame[ 57: -71 ]
# frame		=	frame[ 56: ]
# frame			=	frame[ 58: -70 ]
frame			=	frame[ 58: -70 -4*8 ]

STATE_CHANNELS	=	4

x,y,_			=	frame.shape	#->	 x 400
# RES_HORIZONTAL	=	x
# RES_VERTICAL	=	y
RES_HORIZONTAL	=	x	/	4	/	1					#	pipes move 4 pixels per frame
RES_VERTICAL	=	y	/	2	/	1		#	->	200	#	the bird moves vertically, 10 when jumping
														#	Seems like if it doesn't just stall,
														#	minimum movement is 2 and at least generally
														#	with any movement %2 == 0
														#	Also, everything is painted in pixel-art style
print	RES_HORIZONTAL , '*'	,	RES_VERTICAL
# batch_size			=	128
frame_mean			=	-100000	#	if you want to compute it on the first loaded batch
frame_st_deviation	=	-100000	#	if you want to compute it on the first loaded batch
# frame_mean			=	0.571706774615	#	values that my flawless net got trained with based on its first batch
# frame_st_deviation	=	1.0674802648	#	values that my flawless net got trained with based on its first batch



def preprocess( frame ):
	global	RES_HORIZONTAL
	global	RES_VERTICAL
	global	STATE_CHANNELS
	global	adjust_mean_and_variation
	#	Colors inside frame are fucked up like this:
	#
	# frame	=	frame[ 58: -70 ]	#	some parts of our view are irrelevant
	frame	=	frame[ 58: -70 -4*8 ]	#	some parts of our view are irrelevant
	frame	=	skimage.transform.resize(
											frame
										,	(RES_HORIZONTAL , RES_VERTICAL)
										,	mode='constant'
										)
	# frame	 =	np.delete(frame	,	[1,2]	,	axis=2)		#	red only
	# frame	 =	np.delete(frame	,	[0,2]	,	axis=2)		#	green only
	if		STATE_CHANNELS	==	4:
		frame	 =	np.delete(frame	,	[2]	,	axis=2)		#	we don't need blue
	elif	STATE_CHANNELS	==	2:
		frame[ : , : , 0 ]	-=	frame[ : , : , 1 ]	#	combine into a single channel without loosing clarity on colors
		frame	 =	np.delete(frame	,	[1,2]	,	axis=2)	#	we don't need blue

	frame	*=	4	#	This is probably unnecessary, but that's something I did for historical (of code) reasons.
					#	If STATE_CHANNELS	==	2 then variance should get up, closer to 1, which might be good.
					#	Best practice known to me from supervised learning on images:
					#	variance -> 1, mean value -> 0.
					#	We're getting closer to this, I think (if STATE_CHANNELS	==	2).
					#	But I can't say for a fact if it does make a difference for better.

	frame	 =	frame.reshape(	1	,	RES_HORIZONTAL	,	RES_VERTICAL	,	STATE_CHANNELS / 2	)


	if	adjust_mean_and_variation:
		frame	-=	frame_mean
		frame	*=	1 / frame_st_deviation
	return	np.float16( frame )





batch_size					=	64
# batch_size				=	256
# batch_size				=	128



maximum_loss_threshold		=	0.001

random_move_amount			=	0
random_moves_till_filled_to	=	batch_size	*	16	#	Random probably gives good data,
 													#	but the chance of passing pipes with random

minimum_frame_amount_requirement_for_inclusion			=	20	#	is so low that we can only afford it at the beginning.
minimum_frame_amount_requirement_for_inclusion_later_on	=	30	# -	Set after random batch is collected - upon first recall

number_of_trashy_last_frames_for_starters	=	35	#	There's around 37 frames pipe start -to- pipe start
number_of_trashy_last_frames_later_on		=	37	#	Set after first recall



max_frames_per_run			=	1024	*	2
memory_capacity				=	4096	*	10
# memory_capacity				=	8192	*	1


make_alternative_net		=	False	#	sucked extremely hard for some reason

# per_exp_load_training_epochs	=	1
# per_exp_load_training_epochs	=	10
# training_on_files_decay_rate	=	.97


"""The following values are changed later in the IF code below"""
#	loading memories
how_many_times_to_cycle_training_data	=	1
per_exp_load_training_epochs			=	1

epochs_during_first_recall_training		=	10
total_recall_every_n_lives				=	10
max_times_trained_since_starting_loop	=	1

rnd_chance_multiplied_by	=	1.
disable_exploration			=	False
i_demand_silence			=	False
dont_store_data				=	False
reset_n_retrain_if_failed	=	False
model_to_load				=	False
use_deterministic_action	=	False
do_train_on_directory		=	False
adjust_mean_and_variation	=	False
never_recall				=	False

LEARNING_RATE	=	.1
l2				=	.0001
lr_decay_rate	=	.9995


min_rnd_chance	=	0.00
preset			=	'No preset chosen.'


preset	=	'gather'
# preset	=	'train_on_stored_data_and_gather'
# preset	=	'train_on_data'
# preset	=	'load_model_and_gather'	#	not implemented yet
# preset	=	'load_model_and_test'

print	bg.green	,	preset	,	color.reset

if	(
		preset	==	'gather'
	or
		preset	==	'train_on_stored_data_and_gather'
	):
	rnd_chance_multiplied_by				=	.3		#	random will probably give the best data, so failing fast is not a problem,
	min_rnd_chance							=	0.02	#	it's probably worth the extra diversity of data obtained

	l2										=	0.0001
	lr_decay_rate							=	.998
	#	recall
	epochs_during_first_recall_training		=	10
	total_recall_every_n_lives				=	500
	max_times_trained_since_starting_loop	=	1
	# i_demand_silence						=	True

	if	preset	==	'train_on_stored_data_and_gather':
		#	load data and explore with a bot based on that data
		disable_exploration						=	True
		do_train_on_directory					=	True
		# never_recall							=	True
		# lr_decay_rate							=	.95
		lr_decay_rate							=	.992

		epochs_during_first_recall_training		=	1

		how_many_times_to_cycle_training_data	=	1
		per_exp_load_training_epochs			=	1

elif(
		preset	==	'train_on_data'
	):
	#	loading memories
	how_many_times_to_cycle_training_data	=	2
	per_exp_load_training_epochs			=	1
	l2										=	0.0001
	# l2										=	0.00002
	# lr_decay_rate							=	.975
	lr_decay_rate							=	.95
	# adjust_mean_and_variation				=	True
	never_recall							=	True
	disable_exploration						=	True
	use_deterministic_action				=	True
	dont_store_data							=	True
	do_train_on_directory					=	True
	i_demand_silence						=	True
	reset_n_retrain_if_failed				=	True

elif(
		preset	==	'load_model_and_test'
	):
	#	loading a network
	# adjust_mean_and_variation				=	True
	never_recall							=	True
	disable_exploration						=	True
	use_deterministic_action				=	True
	dont_store_data							=	True
	i_demand_silence						=	True
	model_to_load							=	\
					'immortal-models/'				\
					+	'one'	#flawless

data_load_folder							=	'data-step-1/'
data_save_folder							=	'data-step-1/'


# optimizer	=	Nadam(lr = LEARNING_RATE)
optimizer	=	SGD(lr = LEARNING_RATE)
# optimizer	=	SGD(lr = LEARNING_RATE	,	momentum = 0.9)
# optimizer	=	SGD(lr = LEARNING_RATE	,	momentum = 0.9	,	nesterov = True)
# optimizer	=	RMSprop(lr = LEARNING_RATE)
# optimizer	=	RMSprop(lr = LEARNING_RATE, rho = 0.9, epsilon = 0.1)


loss_name						=	'binary_crossentropy'
# loss_name						=	'mean_squared_error'
kernel_init_type				=	'orthogonal'
bias_initializer_type			=	'zeros'

# +	'RERUNS_batch_sizeX16_'				\
model_name	=\
						preset	\
					+	'_rg_x4_2_'	+	str(random_moves_till_filled_to)	+	'_'				\
					+	str(RES_HORIZONTAL) + 'x' + str(RES_VERTICAL) + 'x' + str(STATE_CHANNELS)				\
					+	'flap_' + str(batch_size)	+ '_'		\
					+	'rnd' + str(rnd_chance_multiplied_by)		+'_'	\
					+	'D128_last'											\
					+	str( per_exp_load_training_epochs )								\
					+	'_'											\
					+	str( l2 )								\
					+	'_'											\
					+	str( total_recall_every_n_lives )								\
					+	'_'											\
					+	str( lr_decay_rate )								\
					+	rnd_name											\
					+	str( LEARNING_RATE )								\
					+	loss_name											\
					+	str(batch_size)										\
					+	str(config.gpu_options.per_process_gpu_memory_fraction)


if	(
		not	os.path.isdir(	data_load_folder	)
	and
		do_train_on_directory
	):
	print	data_load_folder	,	"doesn't exist. Mount the volume, or something else, maybe?"
	sys.exit(0)

if	(
		not	os.path.isdir(	data_save_folder	)
	and
		not	dont_store_data
	):
	print	data_save_folder	,	"doesn't exist. Mount the volume, or something else, maybe?"
	sys.exit(0)


total_recall_last_final_times_trained	=	-10000
total_recall_index						=	0
times_trained					=	0
times_trained_experiment_net	=	0
F						=	0
memory_slots_filled		=	0
memory_append_index		=	0
sequences_in_file		=	0
np.set_printoptions(precision=3)



seed_iterator	=	1
def	load_Exp(
				states
			,	outs
			,	directory
			):
	global	seed_iterator
	start = time.time()
	print	bg.blue	,	'Loading memories'
	print	fg.black,	states
	print				outs	,	color.reset

	#	I don't want to load experiences from several processes concurrently
	lock_path	=	directory	+	'/lock'
	if		os.path.exists(	lock_path	):
		print	bg.red	,	'Storage locked, waiting.'	,	color.reset
		while	os.path.exists(	lock_path	):
			time.sleep(0.1)
		print	bg.green	,	'Storage unlocked'		,	color.reset

	file( lock_path , 'w').close()


	# print	'reading arrays'
	i	=	np.load( states	,	allow_pickle=False)
	o	=	np.load( outs	,	allow_pickle=False)



	# o	=	np.delete(	o	,	[1]	,	axis = 1	)
	# o.shape =	( len(o) )

	# i	=	np.delete(	i	,	[1,2]	,	axis = 3	)
	# print	'read arrays in'	,	time.time()	-	start

	# np.random.seed( seed_iterator )
	# i	=	np.random.permutation( i )
	# np.random.seed( seed_iterator )
	# o	=	np.random.permutation( o )

	# i	=	np.float64(	i )
	# o	=	np.float64(	o )

	seed_iterator	+=	1

	silent_Remove(	lock_path	)

	global	frame_mean
	global	frame_st_deviation
	if	(
			adjust_mean_and_variation
		and
			frame_mean	<	-1000
		):
		frame_mean			=	i.mean	( dtype=np.float64 )
		frame_st_deviation	=	i.std	( dtype=np.float64 )
		print	bg.purple	,	frame_mean	,	'mean'	,	frame_st_deviation	,	'st. deviation'	,	color.reset

	if	adjust_mean_and_variation:
		i	-=	frame_mean
		i	*=	1 / frame_st_deviation


	print	bg.blue	,	'Loaded'	,	len(i)	,	'in'	,	time.time()	- start	,	color.reset
	return	i	,	o



def silent_Remove(filename):
	try:
		os.remove(filename)
	except OSError as e: # this would be "except OSError, e:" before Python 2.6
		if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
			raise # re-raise exception if a different error occurred

class Thread_with_Return(Thread):
	def __init__(self, group=None, target=None, name=None,
				args=(), kwargs={}, Verbose=None):
		Thread.__init__(self, group, target, name, args, kwargs, Verbose)
		self._return = None
	def run(self):
		if self._Thread__target is not None:
			self._return = self._Thread__target(*self._Thread__args,
												**self._Thread__kwargs)
	def join(self):
		Thread.join(self)
		return self._return



def lr_Decay_Function(epoch):
	global	lr_decay_rate
	lrate = LEARNING_RATE * lr_decay_rate ** epoch
	lrate = math.fabs(lrate)
	return max	(
					lrate
				,	0.0001
				)

input_File_Decay	=	lr_Decay_Function
# def input_File_Decay( epoch ):
# 	global	LR
# 	global	lr_decay_rate
#
# 	# print	LEARNING_RATE	,	'LEARNING_RATE'
# 	# print	lr_decay_rate	,	'lr_decay_rate'
# 	# print	epoch			,	'epoch'
#
# 	lrate = LEARNING_RATE * lr_decay_rate ** epoch
# 	lrate = math.fabs(lrate)
# 	return lrate




def	set_Exp	(
				states
			,	outputs
			):
	global	positive_memories_states
	global	positive_memories_outputs
	global	memory_slots_filled
	global	memory_append_index
	global	total_recall_mode
	global	random_move_amount
	global	memory_capacity
	random_move_amount			=	0

	positive_memories_states	=	states
	positive_memories_outputs	=	outputs

	memory_append_index	=	0
	memory_capacity		=	len( positive_memories_states )
	memory_slots_filled	=	len( positive_memories_states )

	total_recall_mode			=	True
	print	bg.blue	,	'Set'	,	memory_slots_filled	,	'memories'	,	color.reset


def	train_Model	(
					model
				,	initial_epoch
				,	input_file_epochs
				,	data_states
				,	data_outs
				,	LR		=	0.1
				# ,	decay	=	0.98
				):

	global	batch_size
	global	lr_decay_rate
	decay	=	lr_decay_rate
	start	=	time.time()

	how_many_samples	=	len	(
								data_states
								)

	how_many_samples	-=	how_many_samples % batch_size	#	we don't want to risk having a tiny batch

	model.fit	(
					data_states	[ : how_many_samples ]
				,	[
					data_outs	[ : how_many_samples ]
					]
				,	initial_epoch	=	initial_epoch
				,	epochs			=	initial_epoch	+	input_file_epochs
				,	batch_size		=	batch_size
				,	callbacks		=	[
										LearningRateScheduler( input_File_Decay )
										]
				)
	print		bg.green	,	time.time()	-	start	,	color.reset



def list_Files	(
					foldername
				,	suffix	=	".npy"
				,	fulldir	=	True
				):
	file_list_tmp = os.listdir(foldername)
	# print len(file_list_tmp)
	# print file_list_tmp
	file_list = []
	if fulldir:
		for item in file_list_tmp:
			if item.endswith(suffix):
				# print	item ,	'endswith'
				file_list.append(os.path.join(foldername, item))
	else:
		for item in file_list_tmp:
			if item.endswith(suffix):
				file_list.append(item)
	return file_list



def	train_On_Files	(
						input_files
					,	output_files
					,	directory
					,	input_file_epochs	=	5
					,	per_load_epochs		=	10
					,	lr					=	0.1
					# ,	decay_rate	=	0.995
					):
	global	positive_memories_states
	global	positive_memories_outputs
	global	memory_slots_filled
	global	memory_append_index
	global	decision_only_model
	# global	decision_only_model
	# global	nn_model
	global	model_name

	# print	input_files
	# loaded_states	=	False
	# loaded_outputs	=	False
	#
	# loading_thread	=	False
	first_run	=	True

	training_epoch	=	0

	input_files_new		=	input_files [:]
	output_files_new	=	output_files[:]

	for	i	in	range(	1	,	input_file_epochs	):
		input_files_new.extend	( input_files	)
		output_files_new.extend	( output_files	)

	input_files		=	input_files_new
	output_files	=	output_files_new

	input_file		=	input_files	[0]
	output_file		=	output_files[0]
	loading_thread	=	Thread_with_Return	(
												target	=	load_Exp
											,	args	=	(
																input_file
															,	output_file
															,	directory
															)
											)
	loading_thread.start()

	plot_model	(
					decision_only_model
				,
					show_shapes	=	True
				,
					to_file	=	(
									"saved-models/decision_model-training-"
								+	model_name
								+	'.png'
								)
				)

	for	i	in	range(	0	,	len(input_files)	):
		loaded_states	,	loaded_outputs	=	loading_thread.join()
		set_Exp	(
					loaded_states
				,	loaded_outputs
				)

		if	(	# if not last
				i
			<	len( input_files )	-1
			):
			input_file		=	input_files	[	i + 1	]
			output_file		=	output_files[	i + 1	]

			loading_thread	=	Thread_with_Return	(
														target	=	load_Exp
													,	args	=	(
																		input_file
																	,	output_file
																	,	directory
																	)
													)
			loading_thread.start()

		train_Model	(
						decision_only_model
					,	training_epoch
					,	per_load_epochs
					,	positive_memories_states	[ : memory_slots_filled ]
					,	positive_memories_outputs	[ : memory_slots_filled	]
					,	lr
					)
		training_epoch	+=	per_load_epochs

		decision_only_model.save	(
										"saved-models/decision_model-training-"
									+	str(lr)
									+	model_name
									)

		print	'Stored model as saved-models/training-' + str(lr) + model_name

	memory_append_index	=	0



def	train_On_Directory	(
							directory
						,	epochs		=	1
						,	sub_epochs	=	1
						):
	print	directory
	silent_Remove(	directory	+	'/lock')

	data_states	=	sorted	(
							list_Files	(
											directory
										,	'states.npy'
										)
							)

	# print	data_states
	data_outputs=	sorted	(
							list_Files	(
											directory
										,	'outputs.npy'
										)
							)

	train_On_Files	(
						data_states
					,	data_outputs
					,	directory
					,	epochs
					,	sub_epochs
					)



def load_Model(	path ):
	model	=	keras.models.load_model(path)
	plot_model	(
					model
				,
					show_shapes	=	True
				,
					to_file	=	(
								'loaded-model.png'
								)
				)
	print	'Saved visualization in loaded-model.png'
	return	model



def	BN( x ):
	return	BatchNormalization( momentum=0.91 )	(x)

def	CONV(
			x
		,	y	=	(3,3)
		,	pad	=	True
		):
	return	Conv2D	(
						x
						# *	2
					,	kernel_size 		=	y
					,	strides				=	(1,1)
					,	activation			=	'relu'
					,	bias_initializer 	=	bias_initializer_type
					,	kernel_initializer	=	kernel_init_type
					,	kernel_regularizer	=	regularizers.l2( l2 )
					,	padding				=	'same'	if	pad	else	'valid'
					)

def	CONV3(
			x
		,	y	=	(3,3,1)
		,	z	=	(1,1,1)
		,	pad	=	True
		):
	return	Conv3D	(
						x
					,	kernel_size 		=	y
					,	strides				=	z
					,	activation			=	'relu'
					,	bias_initializer 	=	bias_initializer_type
					,	kernel_initializer	=	kernel_init_type
					,	kernel_regularizer	=	regularizers.l2( l2 )
					,	padding				=	'same'	if	pad	else	'valid'
					)

def	MAX_POOL( x ):
	return	MaxPooling2D(
							pool_size	=	(2 , 2)
						,	strides		=	(2 , 2)
						,	padding		=	'same'
						# ,	data_format	=	None
						)	(x)


def build_Model():
	print("Model buliding begins")

	# keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=None)

	S	=	Input(shape = (RES_HORIZONTAL, RES_VERTICAL, STATE_CHANNELS, ), name = 'Input')
	# S1	=	MaxPooling2D(
	#
	# 						pool_size=(2, 2), strides=(2,2)
	# 						# pool_size=(4, 4), strides=(4,4)
	#  					,	padding='same', data_format=None
	# 					)		(S)

	# h0	=	AveragePooling2D(
	# h0	=	MaxPooling2D(
	#
	# 						pool_size=(2, 2), strides=(2,2)
	# 						# pool_size=(4, 4), strides=(4,4)
	#  					,	padding='same', data_format=None
	# 					)		(S)

	# flat	=	ResnetBuilder.build_resnet_18( (STATE_CHANNELS, RES_HORIZONTAL, RES_VERTICAL, ), S)
	global	make_alternative_net
	global	l2
	if	make_alternative_net:
		# flat	=	ResnetBuilder.build(
		# 									 (STATE_CHANNELS, RES_HORIZONTAL, RES_VERTICAL, )
		# 								,	S
		# 								,	'basic_block'
		# 								,	[1, 2, 2]
		# 								)
		pass	#	your model here
	else:
	# (input_shape, input_layer, basic_block, [2, 2, 2, 2])
	#

		#	40 x 200

		# PC	=	BN					( S )
		PC	=	CONV	(
							8
						)			( S  )
						# )			( PC  )

		# PC	=	Reshape	(
		# 					(
		# 						RES_HORIZONTAL
		# 					,	RES_VERTICAL
		# 					,	2
		# 					,	STATE_CHANNELS/2
		# 					)
		# 				)			( S )
		# 				# )			( PC )
		# PC	=	CONV3	(
		# 					8
		# 				,	(3,3,1)
		# 				,	(1,1,1)
		# 				)			( PC )
		#
		# PC	=	Reshape	(
		# 					(
		# 						RES_HORIZONTAL
		# 					,	RES_VERTICAL
		# 					,	16#-1
		# 					)
		# 				)			( PC )

		PC	=	MAX_POOL			( PC )
		PC	=	BN					( PC )
		#	/2

		PC	=	CONV	(
							16
						)			( PC )
		PC	=	MAX_POOL			( PC )
		PC	=	BN					( PC )
		#	/4

		PC	=	CONV	(
							16
						)			( PC )
		PC	=	MAX_POOL			( PC )
		PC	=	BN					( PC )
		#	/8

		PC	=	CONV	(
							32
						)			( PC )
		PC	=	MAX_POOL			( PC )
		PC	=	BN					( PC )
		#	/16

		PC	=	CONV	(
							64
						,	(2,3)
						# ,	pad	=	False
						)			( PC )
		PC	=	MAX_POOL			( PC )
		#	/32

		PC	=	BN					( PC )
		PC	=	CONV	(
							128
						,	(1,3)
						)			( PC )
		PC	=	MAX_POOL			( PC )
		#	/64

		#	1 x 5	->	1 x 3	->	1x1


		P	=	Flatten()	( PC )
		P	=	BN			( P  )
		P	=	Dense	(
							128
						,	activation			=	'relu'
						,	kernel_initializer	=	kernel_init_type
						,	kernel_regularizer	=	regularizers.l2( l2 )
						,	bias_initializer	=	bias_initializer_type
						)													(P)

	P	=	BN		( P )
	P	=	Dense	(
						1
					,	name				=	'models_decision'
					,	activation			=	'sigmoid'
					,	kernel_initializer	=	kernel_init_type
					,	kernel_regularizer	=	regularizers.l2( l2 )
					,	bias_initializer	=	bias_initializer_type
					)	(P)






	SE	=	Input(shape = (RES_HORIZONTAL, RES_VERTICAL, STATE_CHANNELS, ), name = 'Input')

	#	probability that we should explore random instead of relying on our answer/prediction
	E	=	AveragePooling2D(
							pool_size=(2, 1), strides=(2,1)
	 					,	padding='same', data_format=None
						# )	(S1)
						)	(SE)

	# E	=	BN					( E )
	#	/2

	E	=	CONV	(
						8
					)			( E )
	E	=	MAX_POOL			( E )
	E	=	BN					( E )

	E	=	CONV	(
						8
					)			( E )
	E	=	MAX_POOL			( E )
	E	=	BN					( E )

	E	=	CONV	(
						16
					)			( E )
	E	=	MAX_POOL			( E )
	E	=	BN					( E )


	# E	=	CONV	(
	# 					16
	# 				,	(2,3)
	# 				)			( E )
	# E	=	MAX_POOL			( E )
	# E	=	BN					( E )


	E	=	Flatten()															(E)
	# E	=	MaxoutDense(	32, nb_feature=2, init=kernel_init_type, bias=True	)	(E)
	E	=	Dense	(
						16
					,	activation			= 'relu'
					,	kernel_initializer	= kernel_init_type
					,	bias_initializer	= bias_initializer_type
					)															(E)
	E	=	BN		( E )
	E	=	Dense	(
						1
					,	name				=	'exploration_neuron'
					,	activation			=	'sigmoid'
					,	kernel_initializer	=	kernel_init_type
					,	bias_initializer	=	bias_initializer_type
					)															(E)


	decision_only_model = Model	(
									inputs	=	S
								,	outputs = 	[
													P
												]
								)

	experiment_only_model = Model	(
										inputs	=	SE
									,	outputs = 	[
														E
													]
									)


	global	optimizer

	decision_only_model.compile		(
										loss = 	{
												'models_decision'		:	loss_name
												}
									,	optimizer	=	optimizer
									)

	experiment_only_model.compile	(
										loss = 	{
												'exploration_neuron'	:	loss_name
												}
									,	optimizer	=	optimizer
									)

	return	decision_only_model	,	experiment_only_model





class Actor():
	def __init__(
				self
				):
		self.poor_bird					=	False
		self.current_frame_this_life	=	0
		self.run_frame_number			=	0
		self.last_state					=	np.zeros	(
															(
																RES_HORIZONTAL
															,	RES_VERTICAL
															,	STATE_CHANNELS
															)
														,
															dtype	=	np.float16
														)

	def play(self):

		global episode_acts
		global episode_rewards
		global episode_states
		global episode_states_large

		global F	#	what frame is it globally?

		global experiment_only_model
		global decision_only_model

		global random_move_amount
		global random_moves_till_filled_to
		global memory_slots_filled
		global episode
		# global starter_episode_amount
		global acted_randomly_store_large
		global rnd_chance_multiplied_by
		global min_rnd_chance
		global i_demand_silence

		global should_explore
		global exploration_enforcement

		# t_start 		=	t
		if	self.poor_bird:
			self.current_frame_this_life	=	0
		self.poor_bird	=	False

		# action_store	=	np.array([])
		action_store	=	[]

		self.run_frame_number	=	0
		while		self.run_frame_number	<	max_frames_per_run	\
				and	not	self.poor_bird	:

			if self.current_frame_this_life	==	0:
				__,__,__	=	game_state.frame_step( [1,0] ) # entirely ignore the starting black screen
				f1,__,__	=	game_state.frame_step( [1,0] ) # do nothing
				f2,__,__	=	game_state.frame_step( [1,0] ) # do nothing
				self.last_state = np.concatenate(
													(
														preprocess	(
																	f2
																	)
													,	preprocess	(
																	f1
																	)
													)
												,
													axis=3
												)

			if	(
					(
							F
						>	random_move_amount
					and
							memory_slots_filled
						>=	random_moves_till_filled_to
					)
				# or
				# 		episode
				# 	<	starter_episode_amount
				):
				with graph.as_default():
					# random_move_likelihood	=	nn_model.predict( self.last_state )
					chance_we_should_better_pick_random	 =	experiment_only_model.predict( self.last_state )[0][0]
					chance_we_should_better_pick_random	*=	rnd_chance_multiplied_by
			else:
				# nn_output	=	[[1],[1]] if np.random.rand()	>	0.9	else [[0],[1]]	#	choose to explore
				chance_we_should_better_pick_random	=	2	#	choose to explore -> chance_we_should_better_pick_random = 100%

			if	disable_exploration:
				chance_we_should_better_pick_random	=	0
			else:
				chance_we_should_better_pick_random	=	max	(
																min_rnd_chance
															,	chance_we_should_better_pick_random
															)

			action	=	-1

			if						np.random.rand()	<	chance_we_should_better_pick_random:
				action_took = 0 if	np.random.rand()	>	0.1		else 1	#	exploring strategy: 10% of the time choose to flap
				acted_randomly_store_large[	self.run_frame_number ]	=	1
			else:
				action	=	decision_only_model.predict( self.last_state )[0][0]
				if	use_deterministic_action:
					action_took = 0 if	0.5					>	action	else 1	#	"stochastic" likelihood-driven choice
				else:
					action_took = 0 if	np.random.rand()	>	action	else 1	#	"stochastic" likelihood-driven choice
				acted_randomly_store_large[	self.run_frame_number ]	=	0

			next_frame, __,	self.poor_bird = game_state.frame_step	(
		 																[0,1] if action_took > 0.5 else [1,0]
																	)


			#	store the state and results of processing that state
			episode_states_large[	self.run_frame_number	]	=	self.last_state
			action_store.append( action_took )
			# action_store							=	np.append(action_store, action_took)


			self.last_state = np.append	(
											preprocess	(
														next_frame
														)
										,
											self.last_state[:, :, :, : STATE_CHANNELS / 2 ]
										,
											axis=3
										)
			if	not	i_demand_silence:
				print	"F" , str(F) , "times_trained" , str( times_trained ) ,	str( times_trained_experiment_net ) , "rnd"	\
			 		,	'%.2f' % chance_we_should_better_pick_random	\
			 		,	'	|	'										\
			 		,	'%.2f' % action								 	\
					,	'	|	'										\
					,	'action_took'	,	action_took
					# ,	int( acted_randomly_store_large[ self.run_frame_number ] )		\


			F								+=	1
			self.run_frame_number			+=	1
			self.current_frame_this_life	+=	1


		# episode_rewards	=	np.zeros(
		# episode_acts	=	np.zeros(
		# episode_acts	=	np.empty(
		# 								(
		# 									self.run_frame_number
		# 									# len( action_store )
		# 								,	2
		# 								)
		# 							,
		# 								dtype	=	np.float16
		# 							)

		#	predictions
		# episode_acts	[ 		: 		,	0	]
		episode_acts	=	np.array(
										action_store
									,
										dtype	=	np.float16
									)
		should_explore	=	np.empty(
										(
										self.run_frame_number
										)
									,
										dtype	=	np.float16
									)
		exploration_enforcement	=\
							np.empty(
										(
										self.run_frame_number
										)
									,
										dtype	=	np.float16
									)

		# exploration_enforcement	[		:		,	0	]	=	0
		# episode_rewards	[	-40	:	-30	,	0	]	=	.2	#	we don't care about any rewards for decisions here
		# episode_rewards	[	-40	:	-30	,	0	]	=	.2	#	we don't care about any rewards for decisions here

		#	exploration
		should_explore			[		:	-40 ]	=	0	#	shouldn't be explored as it's now added to memory
		exploration_enforcement	[		:	-40 ]	=	1

		should_explore			[	-40	:		]	=	1	#	possibly bad moves. Need exploration
		exploration_enforcement	[	-40	:	-30	]	=	.2
		exploration_enforcement	[	-30	:	-20	]	=	.7
		exploration_enforcement	[	-20	:		]	=	1

		# episode_rewards	[		:		,	1	]	/=	100	#	prevent explosion of the exploration network

		# print						np.where(
		#
		# 								acted_randomly_store_large	[
		# 																max	(
		# 																		0
		# 																	,
		# 																			self.run_frame_number
		# 																		-	40
		# 																	)
		# 															:
		# 																self.run_frame_number
		# 								 							]	==	1
		# 									)


		exploration_enforcement	[	#	except we don't want to reinforce that if we simply died during exploration
								np.where(
										acted_randomly_store_large	[
																		max	(
																				0
																			,
																					self.run_frame_number
																				-	40
																			)
																	:
																		self.run_frame_number
																	]	==	1
										)	[0]

										+	max	(
													0
												,
														self.run_frame_number
													-	40
												)
								]	=	0

		if not	self.poor_bird:		#	if we were just limited by frame amount
			exploration_enforcement	[
									-40 :
									]	=	0

		episode_states	=	episode_states_large	[	#	get a view
														:	self.run_frame_number
													]










try:
	start = time.time()

	# GAMMA	=	0.8		#discount value





	positive_memories_states	=	np.zeros(
												(
													memory_capacity
												,	RES_HORIZONTAL,	RES_VERTICAL, STATE_CHANNELS
												)
											,
												dtype	=	np.float16
											)
	episode_states_large		=	np.zeros(
												(
													max_frames_per_run	+	1
													# max_frames_per_run	+	max_frames_per_run	+	1
												,	RES_HORIZONTAL,	RES_VERTICAL, STATE_CHANNELS
												)
											,
												dtype	=	np.float16
											)
	acted_randomly_store_large	=	np.zeros(
												max_frames_per_run	+	1
											,
												dtype	=	np.float16
											)
	positive_memories_outputs	=	np.zeros(
												(
													memory_capacity
												# ,	2
												)
											,
												dtype	=	np.float16
											)


	decision_only_model	,	experiment_only_model	=	build_Model()
	graph	=	tf.get_default_graph()

	if	model_to_load:
		decision_only_model	=	load_Model	(
												model_to_load
											)
		print	'Loaded model'	,	model_to_load

	# nn_model	=	build_Model()
	# # nn_model	=	load_Model( 'saved-models/best_memories_200x72_deep_separated_explorer_0.9_0' )
	#
	# nn_model._make_predict_function()
	#
	# decision_only_model	=	Model	(
	# 									inputs	=	nn_model.input
	# 								,	outputs	=	nn_model.get_layer('models_decision').output
	# 								)
	#
	# decision_only_model.compile	(
	# 								loss = 	{
	# 										'models_decision'	:	loss_name
	# 										}
	# 							,	optimizer	=	optimizer
	# 							)
	#
	# experiment_only_model	=	Model	(
	# 										inputs	=	nn_model.input
	# 									,	outputs	=	nn_model.get_layer('exploration_neuron').output
	# 									)
	#
	# experiment_only_model.compile(
	# 								loss = 	{
	# 										'exploration_neuron'	:	loss_name
	# 										}
	# 							,	optimizer	=	optimizer
	# 							)






	if	do_train_on_directory:
		train_On_Directory	(
								data_load_folder
							,	how_many_times_to_cycle_training_data
							,	per_exp_load_training_epochs
							)




	actor	=	Actor	()



	callbacks_list = 	[
							LearningRateScheduler( lr_Decay_Function )
						# ,	TensorBoard(log_dir='/tmp/' + model_name)
						]


	should_explore			=	[]
	exploration_enforcement	=	[]
	episode_acts			=	[]

	frames_per_lives		=	[]
	total_recall_mode		=	False
	# death_count						=	0
	number_of_trashy_last_frames	=	number_of_trashy_last_frames_for_starters
	#	MAIN FRAME SEQUENCE LOOP
	while True:
		while_time = time.time()


		if	not	total_recall_mode:
			# if	times_trained	==	starter_times_trained_amount +1:
			# 	total_recall_mode	=	True
			# 	print	bg.red	,	'Scheduling a total recall mode soon due to finishing the random series' ,	color.reset

			if	(
							total_recall_last_final_times_trained
						+	total_recall_every_n_lives
					<=	times_trained
				and
					(
							random_moves_till_filled_to
						<=	memory_slots_filled
					or
						disable_exploration
					)
				):
				total_recall_mode	=	True
				print	bg.red	,	'Scheduling total recall mode' ,	color.reset
				number_of_trashy_last_frames					=	number_of_trashy_last_frames_later_on
				minimum_frame_amount_requirement_for_inclusion	=	minimum_frame_amount_requirement_for_inclusion_later_on

		# print	'number_of_trashy_last_frames'	,	number_of_trashy_last_frames
		# print	'disable_exploration'	,	disable_exploration
		# print	'random_moves_till_filled_to'	,	memory_slots_filled

		if	(
				total_recall_mode
			and
				not	never_recall
			):
			if	(
					memory_slots_filled
				>=	random_moves_till_filled_to
				):
				print	bg.red	,	'Entering total recall mode' ,	color.reset
				total_recall_mode	=	False
				# history	=\
				# 		nn_model.fit	(
				# 							positive_memories_states		[ : memory_slots_filled ]
				# 						,	[
				# 								positive_memories_outputs	[ : memory_slots_filled	,	0	]
				# 							,	positive_memories_outputs	[ : memory_slots_filled	,	1	]
				# 							]
				# 						,	initial_epoch	=	times_trained
				# 						,	epochs			=	times_trained + 1
				# 						# ,	epochs			=	times_trained + 1	if	memory_slots_filled	>	256	else	times_trained + 3
				# 												#	we desperately need training on any experience we can get
				# 												#	to move the net from initialization random
				# 						,	batch_size		=	64
				# 						,	callbacks		=	callbacks_list
				# 						)
				how_many_samples	=	len	(
											positive_memories_states[ : memory_slots_filled ]
											)

				how_many_samples	-=	how_many_samples % batch_size	#	we don't want to risk having a tiny batch

				history	=\
						decision_only_model.fit	(
													positive_memories_states		[ : how_many_samples	]
												,	[
														positive_memories_outputs	[ : how_many_samples	]
													]
												,	initial_epoch	=	times_trained
												,	epochs			=	times_trained + 1
												# ,	epochs			=	times_trained + 1	if	memory_slots_filled	>	256	else	times_trained + 3
																		#	we desperately need training on any experience we can get
																		#	to move the net from initialization random
												,	batch_size		=	batch_size
												,	callbacks		=	callbacks_list
												)

				times_trained	+=	1
				times_trained_since_starting_loop	=	0
				while(
						np.mean	(
								history.history['loss']
								)	>	maximum_loss_threshold
					and
						(
								times_trained_since_starting_loop
							<	max_times_trained_since_starting_loop
						or
							(
									total_recall_last_final_times_trained
								<	1
							and
									times_trained_since_starting_loop
								<	epochs_during_first_recall_training
							)
						)
					):
					if	np.mean	(
								history.history['loss']
								)	>	10.:
						print	bg.red	,	'Looks like our network had died'	,	color.reset
						#	might want to just reset all weights and not exit
						sys.exit(0)

					history	=\
							decision_only_model.fit	(
														positive_memories_states		[ : how_many_samples	]
													,	[
															positive_memories_outputs	[ : how_many_samples	]
														]
													,	initial_epoch	=	times_trained
													,	epochs			=	times_trained + 1
													# ,	epochs			=	times_trained + 1	if	memory_slots_filled	>	256	else	times_trained + 3
																			#	we desperately need training on any experience we can get
																			#	to move the net from initialization random
													,	batch_size		=	batch_size
													,	callbacks		=	callbacks_list
													)
					times_trained	+=	1
					times_trained_since_starting_loop	+=	1

					# if	times_trained_since_starting_loop	%	10	==	9:
					# 	print	bg.blue	,	'Permutating'	,	memory_slots_filled	,	color.reset
					# 	np.random.seed( times_trained )
					# 	positive_memories_states[
					# 							: memory_slots_filled
					# 							]=	np.random.permutation	(
					# 														positive_memories_states[
					# 																				: memory_slots_filled
					# 																				]
					# 														)
					# 	np.random.seed( times_trained )
					# 	positive_memories_outputs[
					# 							: memory_slots_filled
					# 							]=	np.random.permutation	(
					# 														positive_memories_outputs[
					# 																				: memory_slots_filled
					# 																				]
					# 														)

				total_recall_last_final_times_trained	=	times_trained


		actor.play()


		if	actor.poor_bird:
			# death_count	+=	1
			frames_per_lives.append( actor.current_frame_this_life )
			# frames_per_lives	=	frames_per_lives[ -20 :  ]
			print	bg.lightgrey
			print	bg.lightgrey
			print	bg.purple				,	np.mean(frames_per_lives[ -20 : ] )	,	color.reset		\
					,	'frames per last'	,	len(	frames_per_lives[ -20 : ] )	,	'lives on avarage'


			print	'Top score:'	,	fg.lightgreen	,	max( frames_per_lives )	,	'frames'	,	color.reset		\

			print	'Last 20 lives lasted'				,	frames_per_lives[-20:]		,	'frames'

			if	reset_n_retrain_if_failed:
				decision_only_model	,	experiment_only_model	=	build_Model()
				if	do_train_on_directory:
					train_On_Directory	(
											data_load_folder
										,	how_many_times_to_cycle_training_data
										,	per_exp_load_training_epochs
										)

		else:
			print	'Frames so far during current life:',	bg.cyan	,	actor.current_frame_this_life	,	color.reset

		print	fg.yellow		,	len( episode_acts )	,	\
				'frames in' , time.time() - while_time	,	color.reset	,	'seconds just got processed'

		print	bg.lightgrey,	fg.black	,	np.mean( frames_per_lives )	,	color.reset		\
				,	'frames per all'		,	bg.purple	,	fg.black	,	len(frames_per_lives),	color.reset		,	'lives on avarage'

		adding_how_many_frames	=	len( episode_acts )		-	number_of_trashy_last_frames

		if	dont_store_data:
			adding_how_many_frames	=	0

		# print	len( episode_acts )	,	'len( episode_acts )'
		# print	number_of_trashy_last_frames	,	'number_of_trashy_last_frames'
		# print	dont_store_data	,	'dont_store_data'
		# print	minimum_frame_amount_requirement_for_inclusion	,	'minimum_frame_amount_requirement_for_inclusion'

		if	(
				adding_how_many_frames
			<	minimum_frame_amount_requirement_for_inclusion
			):	#	we don't want the net to learn only the starting meaningless frames
			# print	adding_how_many_frames
			adding_how_many_frames	=	0

		# print	adding_how_many_frames	,	'adding_how_many_frames'

		# if	memory_slots_filled	>	400:
		# 	if	adding_how_many_frames	<	20:	#	time to get more picky
		# 		adding_how_many_frames	=	0
		#
		if	adding_how_many_frames	>	0:
			jump_events	=	episode_acts[  :  adding_how_many_frames].sum	(
																			dtype	=	np.float64
																			)
			if	(
					jump_events	<	2	#	we don't want the net to learn to just always choose to wait and never jump
				# and
				# 	not	(
				# 			jump_events	>	0		#	enough is enough
				# 		and
				# 				memory_slots_filled	*	5
				# 			<	memory_capacity		#	we can't afford to be so picky at the beginning
				# 		)
				):
				print		'Meh, not adding'				,	adding_how_many_frames	\
						,	'frames, because jumps are mere',	int( jump_events )
				adding_how_many_frames	=	0
			else:
				print		bg.green	,	len	(
												episode_states[ : -number_of_trashy_last_frames ]
												)															\
						,	color.reset	,	'new positive memories formed with'								\
						,	bg.green	,	int ( jump_events	 )											\
						,	color.reset	,	'jumps. memory_append_index ='	,	memory_append_index


		if	adding_how_many_frames	>	0:

			#	going for more-or-less "okay" data to get the engine running
			# if	(
			# 		memory_capacity		#	we can't afford to be so picky at the beginning
			# 	>	memory_slots_filled	*	5
			# 	):
			# 	adding_how_many_frames	+=	10

			# if(
			# 		memory_capacity		#	we can't afford to be so picky at the beginning
			# 	>	memory_slots_filled	*	5
			# 	):
			# 	adding_how_many_frames	+=	5
			sequences_in_file	+=	1

			if	memory_append_index	>=	memory_capacity:
				print	memory_append_index	,	'memory_append_index'
				print	memory_capacity		,	'memory_capacity'
				memory_append_index	=	0
				rnd_name	=	rnd_String()
				print	'Storing positive experiences'			,	rnd_name
				print	'Sum of positive_memories_outputs is '	,	positive_memories_outputs.sum( dtype = np.float64 )
				if	not	dont_store_data:
					try:
						np.save		(
										data_save_folder
										+	model_name + rnd_name + str(memory_capacity)
										+	'-'
										+	str( sequences_in_file )
										+	'_states.npy'
									# ,	np.float16(	positive_memories_states	)
									,	positive_memories_states
									,	allow_pickle	=	False
									)
						np.save		(
										data_save_folder
										+	model_name + rnd_name + str(memory_capacity)
										+	'-'
										+	str( sequences_in_file )
										+	'_outputs.npy'
									# ,	np.float16(	positive_memories_outputs	)
									,	positive_memories_outputs
									,	allow_pickle	=	False
									)
						sequences_in_file	=	0
					except Exception, e:
						print	bg.red	,	'Couldn\'t store. Not enough storage?..' ,	color.reset
						print 'type is:', e.__class__.__name__
						print_exc()
					print	bg.green	,	fg.blue	,	'Stored positive experiences' ,	color.reset
					total_recall_mode	=	True
					print	bg.red	,	'Entering total recall mode after collecting a full batch of positive experience'\
							,	color.reset
				continue


			if	(
						memory_append_index
					+	adding_how_many_frames
				>=
					memory_capacity
				):
				positive_memories_states[
											- adding_how_many_frames
										:
										]	=				\
											episode_states	[
																:	adding_how_many_frames
															]
				positive_memories_outputs[
											- adding_how_many_frames
										:
										]	=				\
											episode_acts	[
																:	adding_how_many_frames
															]

				memory_append_index	=	memory_capacity
				if	not	dont_store_data:
					print	bg.blue	,	'Permutating memorized events'	,	color.reset
					np.random.seed( F )
					positive_memories_states	=	np.random.permutation( positive_memories_states		)
					np.random.seed( F )
					positive_memories_outputs	=	np.random.permutation( positive_memories_outputs	)
			else:
				if	(
							random_moves_till_filled_to
						<=	memory_slots_filled
					and
						np.random.rand()	>	0.1	#	90% chance. We don't need too many starting sequences
					):
					adding_how_many_frames	-=	minimum_frame_amount_requirement_for_inclusion
					es	=	episode_states	[
											minimum_frame_amount_requirement_for_inclusion	:
											]
					ea	=	episode_acts	[
											minimum_frame_amount_requirement_for_inclusion	:
											]
				else:
					es	=	episode_states
					ea	=	episode_acts

				positive_memories_states[
				 							memory_append_index
										:
												memory_append_index
											+	adding_how_many_frames
										]	=	\
											es	[
													:	adding_how_many_frames
												]
				# print	positive_memories_outputs.shape	,	'positive_memories_outputs'
				# print	episode_acts.shape	,	'episode_acts'
				positive_memories_outputs[
				 							memory_append_index
										:
												memory_append_index
											+	adding_how_many_frames
										]	=				\
											ea	[
													:	adding_how_many_frames
												]

				memory_append_index	 +=	 adding_how_many_frames


			memory_slots_filled	=	min	(
											len	(	positive_memories_states	)
										,
											max	(
													memory_slots_filled
												,	memory_append_index
												)
										)


		print	4 , time.time() - while_time	,	'seconds'

		# from_memory_point	=	int	(
		# 								np.random.rand()
		# 							*	memory_slots_filled
		# 							)
		#
		# if	memory_slots_filled	\
		# 	<	batch_size * 1.5:
		# 	from_memory_point	=	0
		#
		# to_memory_point		=	min	(
		# 									from_memory_point
		# 								+	max_frames_per_run
		# 								-	len( episode_acts )
		# 							,
		# 								memory_slots_filled
		# 							)


		print	fg.green	,	memory_slots_filled	,	color.reset			,	'memory_slots_filled - positive memories held'



		if	(
				(
						F
					>	random_move_amount
				and
						memory_slots_filled
					>=	random_moves_till_filled_to
					#	important to learn exploration chance beforehand
				)
				# or
				# 		times_trained
				# 	<	starter_times_trained_amount
			):
			# start_appending_memory	=	time.time()

			# episode_states	=	episode_states	[ : -30 ]	#	so it doesn't die
			# episode_acts	=	episode_acts	[ : -30 ]
			# episode_rewards	=	episode_rewards	[ : -30 ]


			#	LETS TRY WITHOUT
			# episode_states	=	np.append	(
			# 									episode_states
			# 								,	positive_memories_states	[
			# 								 									from_memory_point
			# 																:
			# 																	to_memory_point
			# 																]
			# 								,	axis = 0
			# 								)
			#
			# episode_acts	=	np.append	(
			# 									episode_acts
			# 								,	positive_memories_outputs	[
			# 								 									from_memory_point
			# 																:
			# 																	to_memory_point
			# 																]
			# 								,	axis = 0
			# 								)
			#
			# episode_rewards	=	np.append	(
			# 									episode_rewards
			# 								,	np.ones	(
			# 												(
			# 													to_memory_point - from_memory_point
			# 												,	2
			# 												)
			# 											)
			# 								,	axis = 0
			# 								)
			# episode_rewards	[
			# 						-	(to_memory_point - from_memory_point)
			# 					:
			# 				,
			# 					1
			# 				]	*=	0.2	#	don't discourage exploration too hard



			# print	time.time()		-	start_appending_memory	,	'seconds on appending episode_states & stuff'
			# print	to_memory_point	-	from_memory_point		,	'replay memories used'

			learning_start_time	=	time.time()
			# print	episode_states	[ : max_frames_per_run ].shape
			#
			# how_many_samples	=	len	(
			# 							episode_acts	[ : max_frames_per_run ]
			# 							)	-	5
			#
			# e_states_view	=	episode_states	[ 5	: 5 + how_many_samples ]
			# e_outs_view		=	episode_acts	[ 5	: 5 + how_many_samples ]
			# how_many_samples	=	len( e_states_view )
			# # print	how_many_samples	,	'	-	how_many_samples'
			# how_many_samples	-=	how_many_samples % batch_size	#	we don't want to risk having a tiny batch
			# how_many_samples	=	min	(
			# 								how_many_samples
			# 							,	batch_size	*	4
			# 							)
			# e_states_view	=	e_states_view	[	- how_many_samples	:	]
			# e_outs_view		=	e_outs_view		[	- how_many_samples	:	]
			# print	how_many_samples	,	'	-	how_many_samples'
			how_many_samples	 =	len(	episode_acts	)
			how_many_samples	-=	5
			how_many_samples	-=	how_many_samples % batch_size
			how_many_samples	 =	max	(
											0
										,
											min	(
													how_many_samples
												,	batch_size	*	8
												)
										)
			print	episode_states	[ - how_many_samples	: ].shape

			if	(
						how_many_samples
					>	0
				and
					not disable_exploration
				# and
				# 	0
				):
				# history = experiment_only_model.fit	(
				# 							episode_states		[ : how_many_samples ]
				# 						,	[
				# 								episode_acts	[ : how_many_samples	, 1 ]
				# 							]
				# 						,	initial_epoch	=	times_trained
				# 						,	epochs			=	times_trained + 1
				# 						,	batch_size		=	batch_size
				# 						,	callbacks		=	callbacks_list
				# 						,	sample_weight	=
				# 						 		{
				# 									'exploration_neuron':	episode_rewards[ : how_many_samples , 1 ]
				# 								}
				# 						)
				# print	'episode_acts	[ 20	: how_many_samples	+20	, 1 ]'	,	episode_acts	[ 5	: how_many_samples	+5	, 1 ]
				# print	'episode_rewards[ : how_many_samples	 , 1 ]'	,	episode_rewards[ 	: how_many_samples , 1 ]
				history = experiment_only_model.fit	(
														episode_states	[	- how_many_samples	:	]
													,	[
															episode_acts	[ - how_many_samples :	]
															# episode_acts	[ 5	: how_many_samples	+5	, 1 ]
														]
													,	initial_epoch	=	times_trained
													,	epochs			=	times_trained + 1
													,	batch_size		=	batch_size
													,	callbacks		=	callbacks_list
													,	sample_weight	=
													 		{
																'exploration_neuron':	exploration_enforcement	[ - how_many_samples : ]
															# 'exploration_neuron':	exploration_enforcement[ 5	: how_many_samples	+5	]
															}
													)
				# history = nn_model.fit	(
				# 							episode_states		[ : how_many_samples ]
				# 						,	[
				# 								episode_acts	[ : how_many_samples	, 0 ]
				# 							,	episode_acts	[ : how_many_samples	, 1 ]
				# 							]
				# 						,	initial_epoch	=	times_trained
				# 						,	epochs			=	times_trained + 1
				# 						,	batch_size		=	batch_size
				# 						,	callbacks		=	callbacks_list
				# 						,	sample_weight	=
				# 						 		{
				# 									'models_decision'	:	episode_rewards[ : how_many_samples , 0 ]
				# 								,	'exploration_neuron':	episode_rewards[ : how_many_samples , 1 ]
				# 								}
				# 						# ,	callbacks=[]
				# 						)
			# history = nn_model.fit	(
			# 							episode_states		[ : max_frames_per_run ]
			# 						,	[
			# 								episode_acts	[ : max_frames_per_run	, 0 ]
			# 							,	episode_acts	[ : max_frames_per_run	, 1 ]
			# 							]
			# 						,	initial_epoch	=	times_trained
			# 						,	epochs			=	times_trained + 1
			# 						,	batch_size		=	batch_size
			# 						,	callbacks		=	callbacks_list
			# 						,	sample_weight	=
			# 						 		{
			# 									'models_decision'	:	episode_rewards[ : max_frames_per_run , 0 ]
			# 								,	'exploration_neuron':	episode_rewards[ : max_frames_per_run , 1 ]
			# 								}
			# 						# ,	callbacks=[]
			# 						)
				print	fg.pink	,	'Loss'	,	str( history.history['loss'] )	,	color.reset
				times_trained_experiment_net += 1

				# print	'episode_acts'		,	episode_acts
				# print	'episode_rewards'	,	episode_rewards
				#
				# print	"Learning Rate was" , lr_Decay_Function( times_trained + 1 )						\
				# 		,	model_name	,	'trained in'	,	time.time() - learning_start_time , 'sec'	\
				# 		,	'mean(  (prediction*2-1) * reward  ) ='	,	bg.blue,							\
				# 													np.around(
				# 															np.mean	(
				# 																		(
				# 																				episode_acts	[
				# 																									:	max_frames_per_run
				# 																								,	0
				# 																								]
				# 																				*	2
				# 																			-1
				# 																		)
				# 																	*
				# 																		episode_rewards	[
				# 																							:	max_frames_per_run
				# 																						,	0
				# 																						]
				# 																	)
				# 															,	decimals	=	3
				# 															)\
				# 													,	color.reset



		#	mem refresh
		episode_rewards	= []
		episode_acts	= []


		print	1 , time.time() - while_time	,	'seconds'

		if times_trained % 1000 == 2:
			times_trained	+=	1	# to not save twice in some situations
			# nn_model.save(				"saved-models/" 				+ model_name + "_" + str(times_trained) )
			decision_only_model.save(	"saved-models/decision_model-"		+ model_name + "_" + str(times_trained) )
			experiment_only_model.save(	"saved-models/experiment_model-" 	+ model_name + "_" + str(times_trained) )
			print	'Stored model as saved-models/' + model_name


		print	time.time() - start	,	'seconds'	,	time.time() - while_time	,	'seconds'	,	model_name
except KeyboardInterrupt:
# except:
	# nn_model.save(				"saved-models/" 				+ model_name + "_" + str(times_trained) )
	# decision_only_model.save(	"saved-models/decision_mod-"	+ model_name + "_" + str(times_trained) )
	decision_only_model.save(	"saved-models/decision_model-"		+ model_name + "_" + str(times_trained) )
	experiment_only_model.save(	"saved-models/experiment_model-" 	+ model_name + "_" + str(times_trained) )

	plot_model	(
					decision_only_model
				,
					show_shapes	=	True
				,
					to_file	=	(
									"saved-models/decision_model-"
								+	model_name
								+	'.png'
								)
				)

	plot_model	(
					experiment_only_model
				,
					show_shapes	=	True
				,
					to_file	=	(
									"saved-models/experiment_model-"
								+	model_name
								+	'.png'
								)
				)

	print	'Stored model as saved-models/' + model_name
	# np.save		(
	# 				model_name + '_positive_memories_states.npy'
	# 			,	positive_memories_states
	# 			,	allow_pickle	=	True
	# 			)
	# np.save		(
	# 				model_name + '_positive_memories_outputs.npy'
	# 			,	positive_memories_outputs
	# 			,	allow_pickle	=	True
	# 			)
	print('Bye!')
