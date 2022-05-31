import http.server
import socketserver
import pickle
import pandas as pd
from http.server import BaseHTTPRequestHandler,HTTPServer
import sys
import os

path = os.getcwd() ### Current directory
sys.path.append(path)    #The path were 2code files [new_server.py and Data_analysis.py] are kept
from Data_analysis import create_features    #importing create_features() from the file


########### SERVER STARTING AND HITTING CODE
PORT = 8080

def model_prediction_by_inject(data):
	result = data
	print("USER PAYLOAD",result)
	data = {"payload":[result]}   #storing user payload into a dictionary
	df = pd.DataFrame(data)  #converting the dict into dataframe
	X_test = create_features(df)  #passing the payload to extract features
	final_result = classifier.predict(X_test)  #predicting using model 
	return 'MALICIOUS' if final_result > 0 else 'NOT_MALICIOUS' #output

class myHandler(BaseHTTPRequestHandler):    #server start and listen in initialized port
	
	#Handler for the GET requestsj
	def do_GET(self):
		print(self.path)
		self.send_response(200)
		self.send_header('Content-type','text/html')
		self.end_headers()

		
		# Send the html message
		self.wfile.write(bytes(model_prediction_by_inject(self.path[1:]),"utf8"))    #taking the user payload
		return

# with socketserver.TCPServer(("", PORT), Handler) as httpd:
#     print("serving at port", PORT)
#     httpd.serve_forever()
try :
    print("SERVER HOSTING ...")
    classifier = pickle.load( open(path + "/svm_model.p", "rb")) #loading  svm model
    # classifier = pickle.load( open(path + "/data/tfidf_2grams_randomforest.p", "rb"))
    server = HTTPServer(('', PORT), myHandler)  #starting server
    print("serving at port", PORT)
    server.serve_forever()
except KeyboardInterrupt:
	print ('^C received, shutting down the web server')
	server.socket.close()


