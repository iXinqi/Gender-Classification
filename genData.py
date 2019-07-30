import shutil

# 读取文件中图片信息根据性别分类图片到对应目录中
dirroot = "D:\\性别分类_新数据"
f = open(dirroot+"\\attr_label.txt","r")
i = 0

for line in f.readlines():
    line = line.split(',')
    print(line[21])
#    break
    imgName = line[0]
    if i > 0 and i < 28014:
        if line[21] == '0':
            print("female")
            shutil.move(dirroot+'\\images\\'+imgName, "data\\train\\female\\"+imgName)
        #     移动图片到female目录
        elif line[21]== '1':
            print("male")
            shutil.move(dirroot+'\\images\\'+imgName, "data\\train\\male\\"+imgName)
        #     移动图片到male目录
    else:
        if line[21] == '0':
            print("female")
            shutil.move(dirroot+'\\images\\'+imgName, "data\\test\\female\\"+imgName)
        #     移动图片到female目录
        elif line[21]== '1':
            print("male")
            shutil.move(dirroot+'\\images\\'+imgName, "data\\test\\male\\"+imgName)
            # 未识别男女
    i += 1
f.close()