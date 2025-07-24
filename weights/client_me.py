import requests
#提交文件格式，文件序号\t识别结果
#1	aaaaaa
#2	bbbbbb
#3	CCCCCC
f = open(r"result.txt","rb")#提交的结果以自己的学号命名
files = {'file': f}
r = requests.post(url="http://101.34.251.69:5005/detectfile",files=files)#服务器地址不要修改
print(r.text)
