import sys
import os
import numpy as np
import math
import random
from scipy import stats  
import matplotlib
matplotlib.use( 'Agg' )
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
#import sklearn 

from datetime import datetime
startTime = datetime.now()

print("WELCOME TO A QUICK PYTHON PROGRAM.")
print("It will do machine learning stuff with Numbers.. yep. ")


"""
-----------------------------------------------------
NICHOLAS CHASON 
Machine Learning Code 1. - Native Nays. 
-----------------------------------------------------

"""

def plot_image_basic( xlist, ylist, title, xlab, ylab, legend_val, psize, xlog, ylog, yflip , pcounter, cmap_var=''):
	print("Entered Basic Plot Function")
	
	
	if len(xlist) != len(ylist):
		print("ERROR! X and Y DATA LENGTHS ARE DIFFERENT!")
		print("Length: x_data: %g" % len(xlist))
		print("Length: y_data: %g" % len(ylist))
	if len(xlist)==0 or len(ylist)==0:
		print("ERROR: list length is ZERO!")
		print("Length: x_data: %g" % len(xlist))
		print("Length: y_data: %g" % len(ylist))
	else:
		print("Length: x_data: %g" % len(xlist))
		print("Length: y_data: %g" % len(ylist))

	if legend_val != 0:
		pass
	plot_title=" Blank Title "   
	x_axis="Blank X"
	y_axis="Blank Y"
	pointsize = 5
	#figure_name=os.path.expanduser('~/Feb9LSSHW2_plot1_A' +'.png')
	#figure_name=os.path.expanduser('~/Feb9LSSHW2_plot1_A' +'.png')
	#Choose which type of plot you would like: Commented out.
	#sets new plot features from call. 	
	"""
	if True:
		plot_title = title
		x_axis = xlab
		y_axis = ylab
 		pointsize = psize
 	"""
	#plt.scatter(xlist, ylist, s=pointsize, lw=0)

	#plt.title(plot_title)
	#plt.xlabel(x_axis)
	#plt.ylabel(y_axis)
	#plt.yscale("log")
	"""
	if yflip == True:
		try:
			plt.ylim(max(ylist), min(ylist))
		except:
			print("uh.oh.... try except statement. check ylim.")
	if ylog != 0:
		plt.yscale("log", nonposy='clip')
	if xlog != 0:
		plt.xscale("log", nonposy='clip')
	"""
	plt.imshow(xlist, cmap=cmap_var)
	#plt.xlim(min(xlist), max(xlist) )
	figure_name=os.path.expanduser('~/Feb17astroML_plot%s.png' % pcount)
	plt.savefig(figure_name)
	print("Saving plot: %s" % figure_name)
	plt.clf()

	dummy = pcounter + 1
	return dummy

"""
===================
   LOAD DIGITS
===================
"""
digits = load_digits()

"""
#----------------------------------------
#To print the Descritption of load_digits 
#----------------------------------------
#To print the full description of data...

print digits['DESCR']
"""


"""
=====================
 PRINT INITIAL INFO
=====================
"""
max_pixel_value = 16

#Print the Data Keys
print 'Data Dict Keys: ', digits.keys()

#Loading in data
#Including the pixel values for each sample.
digits_data = digits['data']
total_number_of_images = len(digits_data)
#Get a single random image index for print and plot.
max_image_idx = total_number_of_images
rand_image_idx = int(random.random() * max_image_idx)

digits_targetnames = digits['target_names']
digits_target = digits['target']
print '\nSample Data Matrix: Element #', rand_image_idx
print '------------------------'
#Prints digits_data[rand_image_idx]. 
#The pixel row length is currently set to 8. 
length_row = 8  
for idx, value in enumerate(digits_data[rand_image_idx]):
	print ("%d " % int(value)),
	#print " ",
	if ((idx+1)%length_row == 0):
		print '\n',

print 'Possible Target names: ', digits_targetnames
print 'Truth Targets: ', digits_target 
print 'Total Images: ', len(digits_data)


"""
===========================
 BUILD TRAINING / TEST SET
===========================
"""
test_fraction = 0.25
training_fraction = 1. - test_fraction
#Get Random Indexes. 
test_number_of_images = math.floor(total_number_of_images * test_fraction)
test_idxs = random.sample(range(0, total_number_of_images), int(test_number_of_images))
training_number_of_images = math.floor(total_number_of_images * training_fraction)
training_idxs = random.sample(range(0, total_number_of_images), int(training_number_of_images))
print 'Length test    : ', len(test_idxs)
print 'Length training: ', len(training_idxs)
test_training_ratio = test_number_of_images/training_number_of_images
test_total_ratio = test_number_of_images/total_number_of_images
print 'The Test/Training ratio is: ', test_training_ratio
print 'The Test/Total ratio is: ', test_total_ratio


"""
BUILD A SORTED LIST OF LISTS OF training DATA!
"""
#2 - Dimensional. 0-->9 ; [0 -- > matching indexes]
Training_Index_Master_Array = []
#Loop over all possilbe Number Values. {0 --> 9}
for num in range(0, 10):
	num_idxs = []
	for i, idx in enumerate(training_idxs):
		if digits_target[idx] == num:
			num_idxs.append(idx)
	num_idxs.sort()
	Training_Index_Master_Array.append(num_idxs)

"""
BUILD A SORTED LIST OF LISTS OF test DATA!
NOTE: CAUTION! DO NOT ACCESS THESE UNTIL THE END!!!!!!!
"""
#2 - Dimensional. 0-->9 ; [0 -- > matching indexes]
Test_Index_Master_Array = []
#Loop over all possilbe Number Values. {0 --> 9}
for num in range(0, 10):
	num_idxs = []
	for i, idx in enumerate(test_idxs):
		if digits_target[idx] == num:
			num_idxs.append(idx)
	num_idxs.sort()
	Test_Index_Master_Array.append(num_idxs)

#To Access the indexes of the training set matching Truth = index_TIMA
#index_TIMA = 2
#print Training_Index_Master_Array[index_TIMA]

#print("Sample Training Example")
#print digits_data[99]


#Do something 
"""
==========================================================
                Begin Machine Magic
==========================================================
"""
print("Building the Average Set of Numbers from Training Set...")
"""
          -------------------------------
              Build Average Number
          -------------------------------
""" 
#Declare Variables for loops
#Initialize Array for storing a single Average pixel array for a number. 


#Training_Pixels_Master_Array = [[0 for i in range(10)] for y in range(64)]
#pix_vals     = [[1 for x in range(length_row)] for y in range(length_row)]

pix_vals = [1 for x in range(length_row*length_row)]
#print pix_vals
temp_sum    = 0
idx_counter = 0 
pix_val = 0
#  For each number, for each training example matching that number,
#  for each pixel, FIND THE AVERAGE VALUE. 
#Sums over each number
for num in range(0, 10):
	#Sums over each pixel. 
	for pix_idx in range(0, (length_row*length_row)):
		#tracks the sum of the pixel value for each matching image
		#Sums over each matching image
		for i, index in enumerate(Training_Index_Master_Array[num]):
			pix_val = digits_data[index][pix_idx]
			temp_sum += pix_val 
			idx_counter += 1
		#Store Average Value and Clear temporary values.
		avg_pix_val = temp_sum / idx_counter
		idx_counter = 0
		temp_sum    = 0

		pix_vals[pix_idx] = avg_pix_val
	#Was done as a debegging measure. too lazy to remove. 
	if num == 0:
		Training_Pixel_0 = pix_vals[:]
	if num == 1:
		Training_Pixel_1 = pix_vals[:]
	if num == 2:
		Training_Pixel_2 = pix_vals[:]
	if num == 3:
		Training_Pixel_3 = pix_vals[:]
	if num == 4:
		Training_Pixel_4 = pix_vals[:]
	if num == 5:
		Training_Pixel_5 = pix_vals[:]
	if num == 6:
		Training_Pixel_6 = pix_vals[:]
	if num == 7:
		Training_Pixel_7 = pix_vals[:]
	if num == 8:
		Training_Pixel_8 = pix_vals[:]
	if num == 9:
		Training_Pixel_9 = pix_vals[:]
	if num == 10:
		print("ERROR NUMBER SHOULDNT EQUAL 10...")

#SETS THE TRAINING ARRAY. 
Training_Pixels_Master_Array= [ Training_Pixel_0, \
		Training_Pixel_1,  Training_Pixel_2,  Training_Pixel_3, \
		Training_Pixel_4,  Training_Pixel_5,  Training_Pixel_6, \
		Training_Pixel_7,  Training_Pixel_8,  Training_Pixel_9, ]



rand_number_value = int(digits_target[rand_image_idx])
print("Training_Pixels_Master_Array: %d" % int(rand_number_value))
print("================================")

for idx, value in enumerate(Training_Pixels_Master_Array[rand_number_value]):
	print ("%.2f " % value),
	#print " ",
	if ((idx+1)%length_row == 0):
		print '\n',

print("\nTraining_Pixels_Master_Array: %d" % 1)
print("================================")

for idx, value in enumerate(Training_Pixels_Master_Array[1]):
	print ("%.2f " % value),
	#print " ",
	if ((idx+1)%length_row == 0):
		print '\n',


"""
          -------------------------------
             END: Build Average Number
          -------------------------------
""" 
print("Finished Building Average Numbers from Training Data!... Choosing Test. ")

"""
=============================================
*********************************************
   TESTING PORTION OF THE CODE. ENTER HERE. 
*********************************************
=============================================
"""


"""
=====================================
TESTING SINGLE RANDOM DRAW from test
=====================================
"""
#Uncomment to choose a single Choice to evaluate. 
random_test_idx = random.choice(training_idxs)
print("\nSelected index: %s, as random choice." % random_test_idx)

#Compute the cost of random image from averages store in costs
#INITIALIZE COSTS. 
#costs_SI = [0 for x in range(10)] 
costs_SI = []
costs = 0
for i_counter in range(10):
	print Training_Pixels_Master_Array[i_counter]
	for pix_idx in range(length_row*length_row):
		cost_val = abs(float(Training_Pixels_Master_Array[i_counter][pix_idx]) - \
					float(digits_data[rand_image_idx][pix_idx]))
		#print Training_Pixels_Master_Array[i_counter][pix_idx]
		#print cost_val
		costs += cost_val
	print("Cost = ", costs)
	costs_SI.append(costs)
	print "-------------"
	print costs_SI
	costs = 0

print("======================================================")
print("               COSTS CALCULATED! ...")
print("======================================================")
#PRint out the Costs. 
for i in range(10):
	stringv = float(costs_SI[i])
	print "{0} {1:.2f}".format(i, stringv)



"""
------------------------
FIND MIMIMUM COST
------------------------
"""
predicted_value = costs_SI.index(min(costs_SI))

#Find Secondary Value.
current_cost = 9999999
for i, cost in enumerate(costs_SI):
	if i == predicted_value or cost >= current_cost:
		pass
	else:
		current_cost = cost
		secondary_predicted_value = i 



print("WE FIRST PREDICT A VALUE OF: %d" % predicted_value)
print("It may also be a:  %d" % secondary_predicted_value)

print("THE ACTUAL VALUE WAS : %d" % digits_target[rand_image_idx])





#Initialize relevant things
iterations = 0

"""
====================
 PRINT SAMPLE IMAGE
====================
"""
#initial plots generated to 0. 
pcount = 0
#Set the Colormap
color_map_used = plt.get_cmap('autumn')

#Plot 1. A single number. 
#------------------------------------------------------------------
#Generate a random number for the image from 0 ==> max_image_idx
#max_image_idx = total_number_of_images
#rand_image_idx = int(random.random() * max_image_idx)
title_label = "A single number. " 
x_label = "x coordinate"
y_label = "y coordinate"
x_data  = digits['images'][rand_image_idx]
y_data  = []
legend_val = 0
pointsize = 3
yflip = False
ylog = 0
xlog = 0
pcount = plot_image_basic(x_data, y_data, title_label, x_label, y_label, \
         legend_val, pointsize, xlog, ylog,  yflip, pcount, color_map_used)



"""
print("Now Plotting....")

#
===================
PLOTTING GUESSES 
===================
#
pcount = 0
title_label = "" 
x_label = ""
y_label = ""
x_data  = np.linspace(0, len(errorsLIST), num=len(errorsLIST))
y_data  = errorsLIST
legend_val = 0
pointsize = 3
yflip = False
ylog = 0
xlog = 0

pcount = plot_basic(x_data, y_data, title_label, x_label, y_label, \
         legend_val, pointsize, xlog, ylog,  yflip, pcount)

"""


print("Time: ")
print (datetime.now() - startTime)
print("END. ")




#Doop. 