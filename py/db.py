import pymysql

db = pymysql.connect("127.0.0.1", "root", "cfl656359504", "ims")
cursor = db.cursor()

# sql = "select * from face"
#
# cursor.execute(sql)
#
# data = cursor.fetchall()
#
# print(data)
# #
# # db.commit()
# db.close()

cursor.execute("delete from face")
db.commit()
db.close()
