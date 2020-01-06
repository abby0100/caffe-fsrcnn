#train_folder="/root/tools/yuan/data/mydata/720p/orig/"
train_folder="720p/orig/"
#train_folder="720-png/"
test_folder=$train_folder

#ls $train_folder | awk '{printf("%s%s %s%s\n"), train_folder, $1, test_folder, $1}' train_folder=$train_folder test_folder=$test_folder> train.txt
#ls $train_folder | awk '{printf("%s%s %s%s\n"), train_folder, $1, test_folder, $1}' train_folder=$train_folder test_folder=$test_folder> test.txt

ls $train_folder | awk '{printf("%s%s 0\n"), train_folder, $1}' train_folder=$train_folder > train.txt
cp train.txt train_label.txt
ls $test_folder | awk '{printf("%s%s 0\n"), test_folder, $1}' test_folder=$test_folder> test.txt
cp test.txt test_label.txt
