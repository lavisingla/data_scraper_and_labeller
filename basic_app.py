from selenium import webdriver
import time
from selenium.webdriver.chrome.options import Options
import imghdr
import urllib.request as ur
import random
from PIL import Image
import os
import multiprocessing as mp
from threading import Thread
import csv
import cv2
from flask import Flask,render_template,redirect,request,session
app =Flask(__name__)
app.config['SECRET_KEY'] = "labeller"
#from tasks import download_url

def get_urls(driver,keyword,idx,lockimg,lockurl):
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
	    		continue;
	    	except:
	        	break
	    last_height = new_height

	urls = driver.find_elements_by_xpath("//div[@jscontroller='Q7Rsec']/a/img")
	print("retrieved=="+str(len(urls)))
	url_list =[]
	for url in urls:

		if url==None:
			continue

		try:
			if url.get_attribute("src") == None:
				pass
			else:
				url_list.append(url.get_attribute("src"))
		except:
			pass
	#for url in url_list:
	#	download_url.async_task(queue='url')

	write_csv(url_list,keyword,idx,lockimg,lockurl)


def scrape_initial(keyword,data,query,lockimg,lockurl):
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
	scrape_urls(driver,keyword,"",data,idx,dict_keyword,keyword_queue,count,lockimg,lockurl)

def scrape_urls(driver,keyword,additional_keyword,data_type,idx,dict_keyword,keyword_queue,count,lockimg,lockurl):
	if additional_keyword == keyword:
		count+=1
		scrape_urls(driver,keyword,keyword_queue[count-1],data_type,idx,dict_keyword,keyword_queue,count,lockimg,lockurl)
		return
	if len(keyword_queue) == count:
		driver.close()
		return

	url= "https://www.images.google.com/"
	driver.get(url)
	driver.find_element_by_xpath("//*[@id='sbtc']/div/div[2]/input").send_keys(keyword+" "+additional_keyword)
	driver.find_element_by_xpath("//*[@id='sbtc']/button").click()
	related = driver.find_elements_by_xpath("//a[@class='dgdd6c VM9Z5b']/div")
	get_urls(driver,keyword,idx,lockimg,lockurl)
	for r in related:
		if r.text in dict_keyword and dict_keyword[r.text] == 1:
			continue

		dict_keyword[r.text] = 1
		keyword_queue.append(r.text)
	count+=1
	scrape_urls(driver,keyword,keyword_queue[count-1],data_type,idx,dict_keyword,keyword_queue,count,lockimg,lockurl)


def scrape_urlHelper(query,data_type,lockimg,lockurl):
	threads=[]
	for i in range(len(query)):
		f = open("urls/"+query[i]+".csv",'w')
		f.close()
		print("Thread scrape created!"+ str(i))
		threads.append(Thread(target=scrape_initial,args=(query[i],data_type[i],query,lockimg,lockurl,)))
	for t in threads:
		t.start()
	for r in threads:
		t.join()

def write_imgpath(keyword,idx,paths,lockimg,lockurl):
	if lockimg[idx] == 1:
		while lockimg[idx] ==1 :
			print("write img sleep!")
			time.sleep(6)
	lines=[]
	#with open("file_paths/"+keyword+".csv",'r'):
	#	lines = f.readlines()

	lockimg[idx]=1
	print("writing imgpath!")
	with open("file_paths/"+keyword+".csv",'a') as f:
		writer = csv.writer(f,delimiter=' ',lineterminator='\r')
		for path in paths:
			writer.writerow([path])
	lockimg[idx]=0

def read_imgpath(keyword,count):
	with open("file_paths/"+keyword+".csv",'r')as f:
		lines = f.readlines()

	if len(lines) == 0:
		time.sleep(10)
		return ""
	else:
		print("read image path -- > "+ lines[count][:-1])
		return lines[count][:-1]
	


def write_csv(urls,keyword,idx,lockimg,lockurl):
	print("urls to be written == "+str(len(urls)))
	if lockurl[idx]==1:
		while lockurl[idx] == 1:
			print("write csv sleep")
			time.sleep(5)
	while True:
		lines=[]
		with open("urls/"+keyword+".csv",'r') as f:
			lines = len(f.readlines())
		if lines > 100:
				time.sleep(15)
				continue;
		else:
				break
	lockurl[idx]=1
	print("writitng url csv!")
	with open("urls/"+keyword+".csv",'a') as f:
		writer = csv.writer(f, delimiter=' ',lineterminator='\r')
		for url in urls:
			writer.writerow([url])

	lockurl[idx]=0

def read_csv(keyword,idx,lockimg,lockurl):
	if lockurl[idx]==1:
		while lockurl[idx] == 1:
			print("read csv sleep")
			time.sleep(5)
	lockurl[idx]=1
	lines=[]
	with open("urls/"+keyword+".csv",'r') as f:
			lines = f.readlines()
			print("lines read --> "+str(len(lines)))
			time.sleep(4)
	with open("urls/"+keyword+".csv",'w+') as f:
			pass
	with open("urls/"+keyword+".csv",'w') as f:
			if len(lines) ==0:
				lockurl[idx]=0
				return []
			elif len(lines) >=20:
				read = lines[:20]
				for i in range(len(read)):
					read[i]=read[i][:-1]

				l = lines[20:]
				writer=csv.writer(f,delimiter=' ',lineterminator='\r')
				for line in l:
					line = line[:-1]
					while(True):
						print(line)
						if line[0] == '"':
							line = line[1:-1]
						else:
							break
					writer.writerow([line])
				lockurl[idx]=0
				return read
			else:
				lockurl[idx]=0
				return lines


def convert_to_jpg(filename):
	try:
		temp=filename
		ext = filename.split(".")[-1]
		filename = filename.split(".")[0]+".jpeg"

		if ext == "gif":
			im = Image.open(temp)
			mypallete = im.getpallete()
			im.putpallete(mypallete)
			new_im = Image.new("RGB",im.size)
			new_im.paste(im)
		elif ext=="jpg":
			new_im = Image.open(temp)
		elif ext=='png':
			im = Image.open(temp)
			new_im = Image.new('RGB',im.size)
			new_im.paste(im)
		else:
			os.remove(temp)
			return

		os.remove(temp)
		new_im.save(filename)
	except:
		os.remove(temp)

def download_file(keyword,data_type,idx,lockimg,lockurl,count=0):
	name=""
	paths=[]
	urls = read_csv(keyword,idx,lockimg,lockurl)
	print("lenght of url recieved == "+str(len(urls)))
	if len(urls) == 0:
		time.sleep(10)
	else:
		for url in urls:
			num = count
			name = keyword+str(num)
			count+=1

			try:
				while(True):
					if url[0] =='"':
						url=url[1:-1]
					else:
						break
				print("Downloading -> "+name)
				ur.urlretrieve(url,"files/{}/{}.jpeg".format(keyword,name))
			except Exception as e:
				print(e)
				count-=1
				continue
			ext = imghdr.what("files/{}/{}.jpeg".format(keyword,name))
			if ext == 'jpeg':
				paths.append(name)
				continue
			else:
				os.remove("files/{}/{}.jpeg".format(keyword,name))
				ur.urlretrieve(url,"files/{}/{}.{}".format(keyword,name,ext))
				convert_to_jpg("files/"+keyword+"/"+name+"."+ext)
				paths.append(name)
		write_imgpath(keyword,idx,paths,lockimg,lockurl)
	download_file(keyword,data_type,idx,lockimg,lockurl,count)


def download_helper(query,data_type,lockimg,lockurl):
	threads=[]
	for i in range(len(query)):
		f = open("file_paths/"+query[i]+".csv",'w')
		f.close()
		print("Thread download created!"+ str(i))
		threads.append(Thread(target=download_file,args=(query[i],data_type[i],query.index(query[i]),lockimg,lockurl,)))
	for t in threads:
		t.start()
	for r in threads:
		t.join()

@app.route("/",methods=['GET','POST'])
def index():
	if request.method == 'GET':
		return render_template("index.html")
	else:
		keyword = request.form['keyword']
		keywords = keyword.split("OR")
		data_type=[]
		query=[]
		total_keys = len(query)	
		for k in keywords:
			query.append(k[0:k.find("data")-1].strip())
			data_type.append(k[k.find("data")+5:].strip())
		lockimg = mp.Array('i',len(query))
		lockurl = mp.Array('i',len(query))

		p1 = mp.Process(target=scrape_urlHelper,args=(query,data_type,lockimg,lockurl,))
		p2 = mp.Process(target=download_helper,args=(query,data_type,lockimg,lockurl,))

		p1.start()
		p2.start()
		c_list=[]
		for q in query:
			c_list.append(0)
		session['count'] = c_list
		session['query'] = query
		time.sleep(90)
		return redirect("/view")
		#view(0,query)
		p1.join()
		p2.join()

@app.route('/view',methods=['GET','POST'])
def view():

	if request.method == 'GET':
			print("GET CALLED")
			for q in session['query']:
				idx = session['query'].index(q)
				path = read_imgpath(q,session['count'][idx])
				val = session['count'][idx]
				print(val)
				session['count'][idx]=1+val
				session.modified=True
				print(session['count'][idx])
				if path == "":
					continue
				else:
					return render_template("view.html",path=path,keyword = q)

	else:
		print(request.form['submit'])
		if "accepted" in request.form['submit']:
			print("ACCEPTED")
			pass
		else:
			print("REJEECTED")
			pass

		for q in session['query']:
				print("POST CALLED")
				idx = session['query'].index(q)
				path = read_imgpath(q,session['count'][idx])
				print(path)
				val = session['count'][idx]
				session['count'][idx] = val+1
				session.modified=True
				if path == "":
					continue
				else:
					print("path == "+path+" value == " + str(session['count'][idx]))
					return render_template("view.html",path=path,keyword = q)




if __name__ =='__main__':
	app.run(debug=True)


























































