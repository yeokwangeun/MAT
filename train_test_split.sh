#!/bin/bash

# Script to download and split the AnimalTrack dataset "videos" into train/test sets

##############################
#EDIT HERE 
# ex) out="/home/n0/mlvu014/MAT/AnimalTrack" (DO NOT INCLUDE "/" at the end!)
# Downloaded videos will be split into test/train sets and saved in $out/[test/train]
out="" 
##############################

# download the video files in "current directory"
wget --header="Host: doc-10-cc-docs.googleusercontent.com" --header="User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36" --header="Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7" --header="Accept-Language: en-US,en;q=0.9" --header="Cookie: AUTH_kli4gc1gg78g9kusfdoteb0n7ulb3f94=04822876658038503568|1685511525000|frbuf6v273qfm4e9lalpim7ulmqngimv" --header="Connection: keep-alive" "https://doc-10-cc-docs.googleusercontent.com/docs/securesc/gsdgdqhq4bgf0a664asg74dkrphq8hme/notu009j8fg6fd2pen9mf8p7qt648p8f/1685511600000/03585332288424208474/04822876658038503568/1oIU9tSmQK1Zd6k8kPqo_vFUIJCg-7YT8?e=download&ax=ADWCPKDbk1Z0mpHVeSs-ufpxNXdWiDb9Sy8aEg20B9AGKvH4B3N8JUKzKRKYD1ErjUwtRgBpiUU498vYpLBez_IFazzFDmUWmR0ILnnP1ur4igOUve5fVYZ4ZjXDejKpbLTsOnz1O9iwwxp18RnYZb_i6W4WVQSvgAgnTmsWMSSVKMtc22fz9z1kqqdAFNb1gYIzWv3-oq5hsjdqTVB2LyiKrwTFOBaLAzicF22lT66phQ-hfXwcYrlsmGXIzUlMmS5AjlcYB-HH3Lq3A3qipHLqbOon3BfQ-5erfjpjIZCOs7keJGM_u6UXk0wh99edE2dlJpXieAnFF-17XehAAG9DfMt2AOSyuHXIkXRWwI1t3A53tfAK6D2JfkPje1YogHkqnjDLPgE16Vv0s_yV2IlrztO9qG68vQfK5pyQuj0POkVbL6D14gHwg2-ma315Ptfv4c2Pnawm6NsAKRudEkLaK5OKA_ma7OzP3Veso-AyNd0fmfnPPH41Pzf8iCW4Spfdsf_-hYWVDFK_xSQk5FI0sjOq5775was1tf-5gyTT6_B5ttnSWkbwLJPJ6HP32jNuZ2KKOyr9-b7Ew4OvpHE-9Ph7dIgLlFVyF-dGja3yFttAEfB3kbpbMTz-gxyEVpHJJNh7NR9UAQzsE9JwEkQyg1CUSye38qiVE8UB3xxKdob4WzA847IXv1H_c8rz-tAn_YF4P-zXxpiX8USrNXMSxmb5a7fh14rLrfbW1U9ZWoChENJ-GymwAk1YirN8jiNqyAVtOr99rpDMOHCXg89Js-v58IkaY4o79KXALJ-ckMTLaTU2hlEI4kJRz1XGioOtBauQNlGCR9Kee52FG4g_Fk6cjEFPeSEM8054i-lwggf0_4qxrEq64OJ8M3D7jVWwiInXH2b3ZBD-5x2s7sdWe36jjoer4vnZyWcep90FfafkWgugC34Jb4Gejs-TDkGhLU7YDh8&uuid=1058e104-abda-4b63-926c-a6e09fbc6480&authuser=0" -c -O 'Whole_AnimalTrack.zip'
mkdir -p $out
unzip Whole_AnimalTrack.zip -d $out

mkdir -p $out/train
mkdir -p $out/test

# Move all the videos in videos_all to videos_train and videos_test
# TRAIN SET 
while read line; do
    echo $line
    filename=$(echo $line | awk -F'.' '{print $1}') # extract filename (remove .mp4) // don't know why basename doesn't work here..
    cp $out/videos_all/$filename.mp4 $out/train/
done < $out/videos_train.txt

# TEST SET 
while read line; do
    echo $line
    filename=$(echo $line | awk -F'.' '{print $1}') # extract filename (remove .mp4) // don't know why basename doesn't work here..
    cp $out/videos_all/$filename.mp4 $out/test/
done < $out/videos_test.txt

