%% crop the im into 256*256

clear;clc;
file_path = 'D:/duhanrong/Unet/my_data/last_use_result/'; % �趨����ͼƬ��Ŀ¼
img_path_list = dir(strcat(file_path, '*.jpg')); % ѡ��׺Ϊ .jpg ��ͼƬ
img_num = length(img_path_list); %���ͼƬ����

for j = 1:img_num 
    image_name = img_path_list(j).name;
    image = imread(strcat(file_path, image_name));
    if ndims(image)==3
    image = rgb2gray(image);
    end
    img_bw = im2bw(image);
    img_fill = imfill(img_bw, 'holes');
    imwrite(img_fill, strcat('D:/duhanrong/Unet/my_data/last_use_imfill/', image_name)); % �����ļ�
end
