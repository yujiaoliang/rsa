'''
生成表格文件

'''
import xlwt
def writeform(x,y,z):
    workbook = xlwt.Workbook(encoding = 'utf-8')
    worksheet = workbook.add_sheet("频谱",cell_overwrite_ok=True)
    for i in range(y):
       worksheet.write(i,z,x[i])
    workbook.save("C:\\Users\\yujiaoliang\\Desktop\\result.xls")

if __name__ == "__main__":
 
    cin = [0.0, 0.0, 0.33333333333333337, 0.0, 9, 0.0, 0]
#    for i in range(7):por
    writeform(cin,7,0)
#print(lyst)
   
   
   
   
   