import pymysql

doc = open("/home/cfl/下载/apache-tomcat-9.0.21/webapps/ims/face.txt", 'w', encoding='utf-8')
db = pymysql.connect("127.0.0.1", "root", "cfl656359504", "ims")
cursor = db.cursor()

sql = "select * from face"

cursor.execute(sql)

data = cursor.fetchall()

print(data, file=doc)

doc.close()
db.close()

