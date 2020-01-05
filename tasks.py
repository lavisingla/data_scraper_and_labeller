from celery import Celery
import urllib.request
#import pandas as pd
import gcsfs
import random
import csv
import os
from flask import session
from google.cloud import storage
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=r"C:\Users\Lavi Singla\Desktop\flask\labellingtool-d18ad3b8a5be.json"

app =Celery("tasks",broker="amqp://localhost//")



#creating label.txt for classification model
def create_label_file(bucket_name,project_id,labels):
	fs = gcsfs.GCSFileSystem(project='labellerr-poc')
	with fs.open("{}/{}/labels.txt".format(bucket_name,project_id),'w') as f :
		for label in labels:
			f.write(label+"\n")
	print("Labels.txt created in {} with project - {}".format(bucket_name,project_id))

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

	#blob.upload_from_filename(src_file)
	#print("File {} uploaded to {}.".format(source_file_name, destination_blob_name))




#move file from temp folder to relevant folder in gcs after reviewing
@app.task
def move_file_in_gcs(bucket_name,project_id,old_folder,new_folder,blob_name):
	storage_client = storage.Client()
	source_bucket = storage_client.get_bucket(bucket_name)
	source_blob = source_bucket.blob(project_id+"/"+old_folder+""/+blob_name)
	dst_blob = destination_bucket.blob(project_id+"/"+new_folder+"/"+blob_name)
	new_blob = source_bucket.copy_blob(
	source_blob, source_bucket, dst_blob)
	source_blob.delete()


#retrieve image from gcs buckets temp folder
@app.task
def retrieve_url_from_gcs(bucket_name,project_id,src_file):
	source_blob_name = project_id+"/"+"data/"+src_file
	storage_client=storage.Client()
	bucket = storage_client.bucket(bucket_name)
	blob = bucket.blob(source_blob_name)
	url_lifetime = 3600 #seconds
	serving_url = blob.generate_signed_url(url_lifetime)
	return serving_url


#list name of item in gcs bucket's folder
def list_blobs(bucket_name,project_id,folder):
	storage_client = storage.Client()
	bucket = storage_client.bucket(bucket_name)
	blobs_all = list(bucket.list_blobs())
	blobs_specific = list(bucket.list_blobs(prefix=project_id+"/"+folder+"/"))
	return blobs_specific


#create new bucket in gcs
def create_bucket(bucket_name):
	bucket_name = bucket_name
	storage_client = storage.Client()
	bucket = storage_client.create_bucket(bucket_name)
	print("Bucket {} created".format(bucket.name))


#create a folder inside a bucket
def create_folder(bucket_name,project_id,folder_name):
	storage_client = storage.Client()
	bucket = storage_client.bucket(bucket_name)
	blob = bucket.blob(project_id+"/"+folder_name+"/")
	blob.upload_from_string('', content_type='application/x-www-form-urlencoded;charset=UTF-8')
	print("Foler created -> " + project_id+"/"+folder_name+".")


#writing data.csv and temp_data.csv . Calling split_test_train
def write_to_datacsv(bucket_name,project_id,filename,label):
	fs = gcsfs.GCSFileSystem(project='labellingtool')
	with fs.open("{}/{}/temp_data.csv".format(bucket_name,project_id),'a') as f:
		#writer = csv.writer(f)
		f.write('gs://{}/{}/{}'.format(bucket_name,project_id,filename)+","+label+"\n")
	with fs.open("{}/{}/temp_data.csv".format(bucket_name,project_id),'r') as f:
		temp_data=f.readlines()
		size = len(temp_data)

	if size == 100:
		#adding temp_data to data.csv
		with fs.open("{}/{}/data.csv".format(bucket_name,project_id),'a') as f:
			for item in temp_data:
				f.write(item)

		#splitting temp_data for classifier into test and train
		#pass it to celery task
		#also converting jpeg to tf_records
		#split_train_test(bucket_name,project_id,60,temp_data) # 60 == training data % #temp_data == temp_ata csv

		#clearing temp_data file for next batch
		with fs.open("{}/{}/temp_data.csv".format(bucket_name,project_id),'w+') as f:
			pass

		#create classification task for celery
					


#delete a file from gcs
def delete_from_gcs(bucket_name,project_id,filename):
	storage_client = storage.Client()
	bucket = storage_client.bucket(bucket_name)
	blob = bucket.blob(project_id+"/"+"temp/"+filename)
	blob.delete()


#need to know if labels.txt can be presemt on cloud or local host only
@app.task
def split_train_test(bucket_name,project_id,train_precent,temp):
	randoms = random.sample(range(0,100),train_precent)
	fs = gcsfs.GCSFileSystem(project='labellerr-poc')
	train = fs.open("{}/{}/train.csv".format(bucket_name,project_id),'w')
	test = fs.open("{}/{}/eval.csv".format(bucket_name,project_id),'w')
	writer_test = csv.writer(test)
	writer_train = csv.writer(train)
	for i in range(0,100):
		if i in randoms:
			writer_train.writerow([temp[i]])
		else:
			writer_test.writerrow([temp[i]])
	writer_test.close()
	writer_train.close()

	#calling jpeg_to_tf_record script from here
	os.system("python -m jpeg_to_tf_record.py \
       --train_csv gs://{}/{}/train.csv \
       --validation_csv gs://{}/{}/eval.csv \
       --labels_file gs://{}/{}/labels.txt \
       --project_id {} \
       --output_dir gs://{}/{}/data_as_tf_records".format(bucket_name,project_id,
	   													 bucket_name,project_id,
														 bucket_name,project_id,
														"labellerr-poc",
														 bucket_name,project_id))

	#call classification model to get trained


def classification_model_training():
	pass
	


