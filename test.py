import numpy as np
import sys
sys.path.append("game/")

import pygame
# import wrapped_flappy_bird_to_watch as game
# import wrapped_flappy_bird_slow as game
import wrapped_flappy_bird as game

import skimage
from skimage import transform, color, exposure

import keras
from keras.models import Sequential, Model
from keras.layers.core import Dense, Flatten, Activation
from keras.layers.convolutional import Convolution2D
from keras.optimizers import RMSprop
import keras.backend as K
from	keras.backend	import	pool2d

import pydot
from keras.utils import plot_model

import tensorflow as tf

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction =	0.1
# config.gpu_options.per_process_gpu_memory_fraction =	0.7
set_session(tf.Session(config=config))

graph		=	tf.get_default_graph()
game_state	=	game.GameState()


frame,__,__		=	game_state.frame_step( [1,0] )
frame			=	frame[ 58: -70 -4*8 ]

x,y,_			=	frame.shape
RES_HORIZONTAL	=	x	/	4	/	1					#	pipes move 4 pixels per frame
RES_VERTICAL	=	y	/	2	/	1		#	->	200	#	the bird moves vertically, 10 when jumping
														#	Seems like if it doesn't just stall,
														#	minimum movement is 2 and (at least nearly)
														#	with any movement %2 == 0
														#	Also, everything is painted in pixel-art style
STATE_CHANNELS	=	4

# frame_mean			=	0.571706774615	#	values that my net was trained with based on its first data file
# frame_st_deviation	=	1.0674802648	#	values that my net was trained with based on its first data file


def load_Model(	path ):
	model	=	keras.models.load_model(path)
	plot_model	(
					model
				,
					show_shapes	=	True
				,
					to_file	=	(
								'tested-model.png'
								)
				)
	print	'Saved model visualization in tested-model.png'
	return	model

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
		frame	 =	np.delete(frame	,	[1,2]	,	axis=2)

	frame	*=	4	#	This is probably unnecessary, but that's something I did for historical (of code) reasons.
					#	If STATE_CHANNELS	==	2 then variance should get up, closer to 1, which might be good.
					#	Best practice known to me from supervised learning on images:
					#	variance -> 1, mean value -> 0.
					#	We're getting closer to this, I think (if STATE_CHANNELS	==	2).
					#	But I can't say for a fact if it does make a difference for better.

	frame	 =	frame.reshape(	1	,	RES_HORIZONTAL	,	RES_VERTICAL	,	STATE_CHANNELS / 2	)


	# frame	-=	frame_mean
	# frame	*=	1 / frame_st_deviation
	return	np.float16( frame )


model	=	load_Model	(
							'immortal-models/'
						# +	'one'
						+	'two'
						)

decision_only_model	=	Model	(
									inputs	=	model.input
								,	outputs	=	model.get_layer('models_decision').output
								)



terminal		=	True
current_move	=	0

try:
	while True:

		if terminal:
			f1,__,__	=	game_state.frame_step( [1,0] )	# do nothing
			f2,__,__	=	game_state.frame_step( [1,0] )
			state = np.concatenate	(
										(
											preprocess	(
														f2
														)
										,	preprocess	(
														f1
														)
										)
									,	axis=3
									)
			terminal = False


		with graph.as_default():
			model_output	=	decision_only_model.predict(state)

		# print	model_output
		action 		=	model_output[0]
		jumping		=	True if action > 0.5 else False	#	top "deterministic" choice

		next_frame, __,	terminal = game_state.frame_step	(
																[0,1] if jumping else [1,0]
															)

		# if	not	terminal:
		# 	next_frame,	__, terminal = game_state.frame_step( [1,0] )

		state = np.append	(
								preprocess	(
											next_frame
											)
							,
								state[:, :, :, : STATE_CHANNELS /2 ]
							,
								axis=3
							)

		current_move	+=	1

		if	current_move	%	50000	==	0:
			print	'Move'	,	move

		if terminal:
			print	'Died =( Move'	,	current_move
			current_move	=	0
except KeyboardInterrupt:
	print('Bye!')
