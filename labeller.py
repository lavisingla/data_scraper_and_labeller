#importing libraries
from selenium import webdriver
import time
from selenium.webdriver.chrome.options import Options
import imghdr
import urllib.request
import random
import datetime
from PIL import Image
import os
import multiprocessing as mp
from threading import Thread
import csv
import time
import gcsfs
import cv2
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model,load_model

import numpy as np
from google.cloud import storage
from flask import Flask,render_template,redirect,request,session


#environment ready
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r'C:\Users\Lavi Singla\Desktop\flask\labellingtool-d18ad3b8a5be.json'
app =Flask(__name__)
app.config['SECRET_KEY'] = "labeller"


################################################################################
#									Classification part

def lr_schedule(epoch):
	"""Learning Rate Schedule

	Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
	Called automatically every epoch as part of callbacks during training.

	# Arguments
	    epoch (int): The number of epochs

	# Returns
	    lr (float32): learning rate
	"""
	lr = 1e-3
	if epoch > 180:
	    lr *= 0.5e-3
	elif epoch > 160:
	    lr *= 1e-3
	elif epoch > 120:
	    lr *= 1e-2
	elif epoch > 80:
	    lr *= 1e-1
	print('Learning rate: ', lr)
	return lr


def resnet_layer(inputs,
	             num_filters=16,
	             kernel_size=3,
	             strides=1,
	             activation='relu',
	             batch_normalization=True,
	             conv_first=True):
	"""2D Convolution-Batch Normalization-Activation stack builder

	# Arguments
	    inputs (tensor): input tensor from input image or previous layer
	    num_filters (int): Conv2D number of filters
	    kernel_size (int): Conv2D square kernel dimensions
	    strides (int): Conv2D square stride dimensions
	    activation (string): activation name
	    batch_normalization (bool): whether to include batch normalization
	    conv_first (bool): conv-bn-activation (True) or
	        bn-activation-conv (False)

	# Returns
	    x (tensor): tensor as input to the next layer
	"""
	conv = Conv2D(num_filters,
	              kernel_size=kernel_size,
	              strides=strides,
	              padding='same',
	              kernel_initializer='he_normal',
	              kernel_regularizer=l2(1e-4))

	x = inputs
	if conv_first:
	    x = conv(x)
	    if batch_normalization:
	        x = BatchNormalization()(x)
	    if activation is not None:
	        x = Activation(activation)(x)
	else:
	    if batch_normalization:
	        x = BatchNormalization()(x)
	    if activation is not None:
	        x = Activation(activation)(x)
	    x = conv(x)
	return x




def resnet_v2(input_shape, depth, num_classes=2):
	"""ResNet Version 2 Model builder [b]

	Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
	bottleneck layer
	First shortcut connection per layer is 1 x 1 Conv2D.
	Second and onwards shortcut connection is identity.
	At the beginning of each stage, the feature map size is halved (downsampled)
	by a convolutional layer with strides=2, while the number of filter maps is
	doubled. Within each stage, the layers have the same number filters and the
	same filter map sizes.
	Features maps sizes:
	conv1  : 32x32,  16
	stage 0: 32x32,  64
	stage 1: 16x16, 128
	stage 2:  8x8,  256

	# Arguments
	    input_shape (tensor): shape of input image tensor
	    depth (int): number of core convolutional layers
	    num_classes (int): number of classes (CIFAR10 has 10)

	# Returns
	    model (Model): Keras model instance
	"""
	if (depth - 2) % 9 != 0:
	    raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
	# Start model definition.
	num_filters_in = 16
	num_res_blocks = int((depth - 2) / 9)

	inputs = Input(shape=input_shape)
	# v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
	x = resnet_layer(inputs=inputs,
	                 num_filters=num_filters_in,
	                 conv_first=True)

	# Instantiate the stack of residual units
	for stage in range(3):
	    for res_block in range(num_res_blocks):
	        activation = 'relu'
	        batch_normalization = True
	        strides = 1
	        if stage == 0:
	            num_filters_out = num_filters_in * 4
	            if res_block == 0:  # first layer and first stage
	                activation = None
	                batch_normalization = False
	        else:
	            num_filters_out = num_filters_in * 2
	            if res_block == 0:  # first layer but not first stage
	                strides = 2    # downsample

	        # bottleneck residual unit
	        y = resnet_layer(inputs=x,
	                         num_filters=num_filters_in,
	                         kernel_size=1,
	                         strides=strides,
	                         activation=activation,
	                         batch_normalization=batch_normalization,
	                         conv_first=False)
	        y = resnet_layer(inputs=y,
	                         num_filters=num_filters_in,
	                         conv_first=False)
	        y = resnet_layer(inputs=y,
	                         num_filters=num_filters_out,
	                         kernel_size=1,
	                         conv_first=False)
	        if res_block == 0:
	            # linear projection residual shortcut connection to match
	            # changed dims
	            x = resnet_layer(inputs=x,
	                             num_filters=num_filters_out,
	                             kernel_size=1,
	                             strides=strides,
	                             activation=None,
	                             batch_normalization=False)
	        x = keras.layers.add([x, y])

	    num_filters_in = num_filters_out

	# Add classifier on top.
	# v2 has BN-ReLU before Pooling
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = AveragePooling2D(pool_size=8)(x)
	y = Flatten()(x)
	outputs = Dense(num_classes,
	                activation='softmax',
	                kernel_initializer='he_normal')(y)

	# Instantiate model.
	model = Model(inputs=inputs, outputs=outputs)
	return model




def classifier(bucket_name,project_id,batch_size,epochs):
	print("classification model started")

	client = storage.Client()
	bucket = client.bucket(bucket_name)
	x_train=[]
	y_train=[]
	x_test=[]
	y_test=[]

	#getting labels.txt
	fs = gcsfs.GCSFileSystem(project='labellingtool')
	with fs.open("{}/{}/labels.txt".format(bucket_name,project_id),'r') as f:
	    classes = f.readlines()
	    labels=[]
	    for elem in classes:
	        labels.append(elem[:-1])
	    num_classes = len(classes)
	print("loaded labels.txt")


	#setting parameters
	data_augmentation = True
	subtract_pixel_mean = True
	version = 2
	n=3
	# Computed depth from supplied model parameter n
	if version == 1:
	    depth = n * 6 + 2
	elif version == 2:
	    depth = n * 9 + 2
	    

	#data preprocessing
	#getting train_label.csv
	fs = gcsfs.GCSFileSystem(project='labellingtool')
	with fs.open("{}/{}/train.csv".format(bucket_name,project_id),'r') as f:
	    data = f.readlines()
	    lines=[]
	    for line in data:
	        split = line.split(",")
	        lines.append([split[0],split[1][:-1]])
	    
	for line in lines:
	    filename = line[0].split("/")[-1]
	    ext = filename.split(".")[-1]
	    blob = bucket.blob(project_id+"/data/"+filename)
	    blob.download_to_filename("temp."+ext)
	    im = cv2.imread("temp."+ext)
	    im = cv2.resize(im,(128,128))
	    x_train.append(im)
	    label = line[1]
	    temp = np.ndarray(shape=(1), dtype=float)
	    temp[0] = labels.index(label)
	    y_train.append(temp)
	    os.remove("temp."+ext)

	#getting test data
	with fs.open("{}/{}/test.csv".format(bucket_name,project_id),'r') as f:
	    data = f.readlines()
	    lines=[]
	    for line in data:
	        split = line.split(",")
	        lines.append([split[0],split[1][:-1]])
	    
	for line in lines:
	    filename = line[0].split("/")[-1]
	    ext = filename.split(".")[-1]
	    blob = bucket.blob(project_id+"/data/"+filename)
	    blob.download_to_filename("temp."+ext)
	    im = cv2.imread("temp."+ext)
	    im = cv2.resize(im,(128,128))
	    x_test.append(im)
	    label = line[1]
	    temp = np.ndarray(shape=(1), dtype=float)
	    temp[0] = labels.index(label)
	    y_test.append(temp)
	    os.remove("temp."+ext)

	    #converting list to numpy array
	    x_train = np.asarray(x_train)
	    x_test = np.asarray(x_test)
	    y_train=np.asarray(y_train)
	    y_test=np.asarray(y_test)
	    print(y_train.shape,y_test.shape,x_train.shape,x_test.shape)
	    time.sleep(10)

	    # Input image dimensions.
	    input_shape = x_train.shape[1:]

	    # Normalize data.
	    x_train = x_train.astype('float32') / 255
	    x_test = x_test.astype('float32') / 255

	    # If subtract pixel mean is enabled
	    if subtract_pixel_mean:
	        x_train_mean = np.mean(x_train, axis=0)
	        x_train -= x_train_mean
	        x_test -= x_train_mean

	    print('x_train shape:', x_train.shape)
	    print(x_train.shape[0], 'train samples')
	    print(x_test.shape[0], 'test samples')
	    print('y_train shape:', y_train.shape)

	    # Convert class vectors to binary class matrices.
	    y_train = keras.utils.to_categorical(y_train, num_classes)
	    y_test = keras.utils.to_categorical(y_test, num_classes)



	    #creating model and loading 
	    blob = bucket.blob(project_id+"/"+"model.h5")
	    if blob.exists():
	    	blob.download_to_filename("model.h5")
	    	model = load_model("model.h5")
	    else:
	    	model = resnet_v2(input_shape=input_shape, depth=depth)

	    model.compile(loss='categorical_crossentropy',
	                optimizer=Adam(lr=lr_schedule(0)),
	                metrics=['accuracy'])

	    model_type = "version2"
	    # Prepare model model saving directory.
	    save_dir = os.path.join(os.getcwd(), 'saved_models')
	    model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
	    if not os.path.isdir(save_dir):
	        os.makedirs(save_dir)
	    filepath = os.path.join(save_dir, model_name)

	    # Prepare callbacks for model saving and for learning rate adjustment.
	    checkpoint = ModelCheckpoint(filepath=filepath,
	                                monitor='val_acc',
	                                verbose=1,
	                                save_best_only=True)

	    lr_scheduler = LearningRateScheduler(lr_schedule)

	    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
	                                cooldown=0,
	                                patience=5,
	                                min_lr=0.5e-6)

	    callbacks = [checkpoint, lr_reducer, lr_scheduler]

	    # Run training, with or without data augmentation.
	    if not data_augmentation:
	        print('Not using data augmentation.')
	        model.fit(x_train, y_train,
	                batch_size=batch_size,
	                epochs=epochs,
	                validation_data=(x_test, y_test),
	                shuffle=True,
	                callbacks=callbacks)
	    else:
	        print('Using real-time data augmentation.')
	        # This will do preprocessing and realtime data augmentation:
	        datagen = ImageDataGenerator(
	            # set input mean to 0 over the dataset
	            featurewise_center=False,
	            # set each sample mean to 0
	            samplewise_center=False,
	            # divide inputs by std of dataset
	            featurewise_std_normalization=False,
	            # divide each input by its std
	            samplewise_std_normalization=False,
	            # apply ZCA whitening
	            zca_whitening=False,
	            # epsilon for ZCA whitening
	            zca_epsilon=1e-06,
	            # randomly rotate images in the range (deg 0 to 180)
	            rotation_range=0,
	            # randomly shift images horizontally
	            width_shift_range=0.1,
	            # randomly shift images vertically
	            height_shift_range=0.1,
	            # set range for random shear
	            shear_range=0.,
	            # set range for random zoom
	            zoom_range=0.,
	            # set range for random channel shifts
	            channel_shift_range=0.,
	            # set mode for filling points outside the input boundaries
	            fill_mode='nearest',
	            # value used for fill_mode = "constant"
	            cval=0.,
	            # randomly flip images
	            horizontal_flip=True,
	            # randomly flip images
	            vertical_flip=False,
	            # set rescaling factor (applied before any other transformation)
	            rescale=None,
	            # set function that will be applied on each input
	            preprocessing_function=None,
	            # image data format, either "channels_first" or "channels_last"
	            data_format=None,
	            # fraction of images reserved for validation (strictly between 0 and 1)
	            validation_split=0.0)

	        # Compute quantities required for featurewise normalization
	        # (std, mean, and principal components if ZCA whitening is applied).
	        datagen.fit(x_train)

	        # Fit the model on the batches generated by datagen.flow().
	        model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
	                            validation_data=(x_test, y_test),steps_per_epoch =50,
	                            epochs=epochs, verbose=1, workers=4,
	                            callbacks=callbacks)
	        model.save_weights("model.h5")
	        blob = bucket.blob(project_id+"/"+"model.h5")
	        blob.upload_from_filename("model.h5")
	        # # Score trained model.
	        # scores = model.evaluate(x_test, y_test, verbose=1)
	        # print('Test loss:', scores[0])
	        # print('Test accuracy:', scores[1])


################################################################################


##################################################################################
# 									GCS FUNCTIONS 


#add a file to gcs buckets folder
def add_to_gcs(bucket_name,project_id,folder,dst_file,url):
	storage_client = storage.Client()
	bucket = storage_client.bucket(bucket_name)
	
	try:
		response = urllib.request.urlretrieve(url)
		format = response[1].get_content_type().split("/")[1]
		response = urllib.request.urlretrieve(url,"test.{}".format(format))
		blob = bucket.blob(project_id+"/"+folder+"/"+dst_file+"."+format)
		blob.upload_from_filename("test.{}".format(format))
		os.remove("test.{}".format(format))
		print("Uploaded image : " + dst_file)
	except Exception as e:
		print(e)
		print("Upload failed !!")

#divide temp.data into test and train
def split_train_test(bucket_name,project_id,train_precent,temp,total):
	randoms = random.sample(range(0,total),int(train_precent/100*total))
	fs = gcsfs.GCSFileSystem(project='labellingtool')
	train = fs.open("{}/{}/train.csv".format(bucket_name,project_id),'w')
	test = fs.open("{}/{}/test.csv".format(bucket_name,project_id),'w')
	
	for i in range(0,total):
		if i in randoms:
			train.write(temp[i])
		else:
			test.write(temp[i])
	train.close()
	test.close()
	print("data splitted!!")

def delete_from_gcs(bucket_name,filename):
	storage_client = storage.Client()
	bucket = storage_client.bucket(bucket_name)
	blob = bucket.blob(filename)
	blob.delete()

#writing data.csv and temp_data.csv . Calling split_test_train
def write_to_datacsv(bucket_name,project_id,filename,label):
	fs = gcsfs.GCSFileSystem(project='labellingtool')
	with fs.open("{}/{}/temp_data.csv".format(bucket_name,project_id),'r') as f:
		temp_data=f.readlines()
	with fs.open("{}/{}/temp_data.csv".format(bucket_name,project_id),'w') as f:
		for data in temp_data:
			f.write(data)
		f.write('gs://{}/{}/{}'.format(bucket_name,project_id,filename)+","+label+"\n")
	with fs.open("{}/{}/temp_data.csv".format(bucket_name,project_id),'r') as f:
		temp_data=f.readlines()
		print("length of temp_dat -> "+str(len(temp_data)))
		size = len(temp_data)

	total=6
	if size == total:
		print("condition for classifier fulfilled!")
		#adding temp_data to data.csv
		with fs.open("{}/{}/data.csv".format(bucket_name,project_id),'a') as f:
			for item in temp_data:
				f.write(item)

		#splitting temp_data for classifier into test and train
		#pass it to celery task
		#also converting jpeg to tf_records
		split_train_test(bucket_name,project_id,50,temp_data,total) # 60 == training data % #temp_data == temp_ata csv
		#clearing temp_data file for next batch
		classifier("labellerr-poc","project_9",2,10)
		with fs.open("{}/{}/temp_data.csv".format(bucket_name,project_id),'w+') as f:
			pass

		#create classification task for celery
					

def retrieve_url_from_gcs(bucket_name,file):
	storage_client=storage.Client()
	bucket = storage_client.bucket(bucket_name)
	blob = bucket.blob(file)
	serving_url = blob.generate_signed_url(expiration=datetime.timedelta(minutes=15),version='v4',method='GET')
	return serving_url


#list name of item in gcs bucket's folder
def list_blobs(bucket_name,project_id,folder):
	storage_client = storage.Client()
	bucket = storage_client.bucket(bucket_name)
	blobs_specific = list(bucket.list_blobs(prefix=project_id+"/"+folder+"/"))
	names=[]
	for i in blobs_specific:
		names.append(i.name)
	return names

#creating label.txt for classification model
def create_file(bucket_name,project_id,labels,name,format):
	fs = gcsfs.GCSFileSystem(project='labellingtool')
	with fs.open("{}/{}/{}.{}".format(bucket_name,project_id,name,format),'w') as f :
		for label in labels:
			f.write(label+"\n")
	print("{}.{} created in {} with project - {}".format(name,format,bucket_name,project_id))

##########################################################################################

##########################################################################################
#							Scraping functionssss

#scroll complete page and collect all urls
def get_urls(driver,keyword,idx,count):
	print("Get_urls called!")
	SCROLL_PAUSE_TIME = 3
	# Get scroll height
	last_height = driver.execute_script("return document.body.scrollHeight")

	while True:
	    # Scroll down to bottom
	    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

	    # Wait to load page
	    time.sleep(SCROLL_PAUSE_TIME)

	    # Calculate new scroll height and compare with last scroll height
	    new_height = driver.execute_script("return document.body.scrollHeight")
	    if new_height == last_height:
	    	try:
	    		driver.find_element_by_xpath("//div[@id='smbw']").click()
	    		continue
	    	except:
	        	break
	    last_height = new_height

	urls = driver.find_elements_by_xpath("//div[@jscontroller='Q7Rsec']/a/img")
	print("retrieved=="+str(len(urls)))

	for url in urls:
		if url==None:
			continue
		try:
			if url.get_attribute("src") == None:
				pass
			else:
				#adding file to gcs temp folder for reviewing 
				name = "project_id"+str(time.time())
				time.sleep(5)
				#add_to_gcs('labellerr-poc','project_9',"data",str(name),url.get_attribute("src"))
				count+=1
		except:
			pass
		
	return count




def scrape_urls(driver,keyword,data_type,idx,dict_keyword,keyword_queue,count):

	if len(keyword_queue) == 0:
		driver.close()
		return
		
	temp = keyword_queue.pop(0)
	if temp  == keyword:
		keyword_queue.append(" ")
		scrape_urls(driver,keyword,data_type,idx,dict_keyword,keyword_queue,count)
		return
	
	url= "https://www.images.google.com/"
	driver.get(url)
	driver.find_element_by_xpath("//*[@id='sbtc']/div/div[2]/input").send_keys(keyword+" "+temp)
	driver.find_element_by_xpath("//*[@id='sbtc']/button").click()
	related = driver.find_elements_by_xpath("//a[@class='dgdd6c VM9Z5b']/div")
	count = get_urls(driver,keyword,idx,count)
	for r in related:
		if r.text in dict_keyword and dict_keyword[r.text] == 1:
			continue
		dict_keyword[r.text] = 1
		keyword_queue.append(r.text)
	scrape_urls(driver,keyword,data_type,idx,dict_keyword,keyword_queue,count)

def scrape_initial(keyword,data,query):
	idx = query.index(keyword)
	name = keyword
	chrome_options = Options()
	chrome_options.add_argument('--ignore-certificate-errors')
	chrome_options.add_argument('--ignore-ssl-errors')
	#chrome_options.add_argument("--headless")
	driver = webdriver.Chrome(options=chrome_options)
	count=0
	dict_keyword={}
	keyword_queue=[]
	keyword_queue.append(name)
	scrape_urls(driver,keyword,data,idx,dict_keyword,keyword_queue,count)

def scrape_urlHelper(query,data_type):
	threads=[]
	scrape_initial(query[0],"img",query)
	'''
	for i in range(len(query)):
		#f = open("urls/"+query[i]+".csv",'w')
		#f.close()
		print("Thread scrape created!"+ str(i))
		threads.append(Thread(target=scrape_initial,args=(query[i],data_type[i],query,)))
	for t in threads:
		t.start()
	for t in threads:
		t.join()'''
#########################################################################################


@app.route("/",methods=['GET','POST'])
def index():
	if request.method == 'GET':
		return render_template("index.html")
	else:
		keyword = request.form['keyword']
		keywords = keyword.split("OR")
		data_type=[]
		query=[]
		for k in keywords:
			query.append(k[0:k.find("data")-1].strip())
			data_type.append(k[k.find("data")+5:].strip())
	
		p1 = mp.Process(target=scrape_urlHelper,args=(query,data_type,))
		create_file("labellerr-poc","project_9",query,"labels","txt")
		create_file("labellerr-poc","project_9",[],"temp_data","csv")
		

		p1.start()
		#session variable

		session['filename'] = []
		session['count']=0
		session['query'] = query
		session['data_relevant'] = []
		print("Going to view!!========================")
		return redirect("/view")
		


@app.route('/view',methods=['GET','POST'])
def view():
	while True:
		if len(session['filename']) == session['count']:
			session['filename'] = list_blobs("labellerr-poc",'project_9','data')
			time.sleep(10)
			continue
		else:
			break


	if request.method == 'GET':
			filename = session['filename'][session['count']] #with extension
			session['count']+=1
			session.modified=True
			url = retrieve_url_from_gcs("labellerr-poc",filename) #getting signed url
			return render_template("view.html",query=session['query'],url=url,filename=filename)


	else:
		return_value = request.form['submit']
		if "accepted" in return_value:
			label = return_value.split(":")[1]
			filename= return_value.split(":")[2]
			print("writing to datacsv!!")
			write_to_datacsv("labellerr-poc","project_9",filename,label) #writing to data.csv
		else:
			print("deleting current image from database")
			delete_from_gcs("labellerr-poc",filename) #deleting from storage
			pass
		filename = session['filename'][session['count']]
		session['count']+=1
		session.modified=True
		url = retrieve_url_from_gcs("labellerr-poc",filename)
		return render_template("view.html",query=session['query'],url=url,filename=filename)



if __name__ =='__main__':
	print("Starting flask application !!")
	app.run(debug=False,threaded=False)


























































