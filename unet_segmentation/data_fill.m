%% crop the im into 256*256

clear;clc;
file_path = 'D:/duhanrong/Unet/my_data/last_use_result/'; % 设定你存放图片的目录
img_path_list = dir(strcat(file_path, '*.jpg')); % 选后缀为 .jpg 的图片
img_num = length(img_path_list); %获得图片数量

for j = 1:img_num 
    image_name = img_path_list(j).name;
    image = imread(strcat(file_path, image_name));
    if ndims(image)==3
    image = rgb2gray(image);
    end
    img_bw = im2bw(image);
    img_fill = imfill(img_bw, 'holes');
    imwrite(img_fill, strcat('D:/duhanrong/Unet/my_data/last_use_imfill/', image_name)); % 保存文件
end
