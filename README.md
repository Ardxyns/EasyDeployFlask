# Easy Deploy Machine Learning WEB with Flask
This repository provide machine learning web app with some ready feature using flask and deploy it in [pythonanywhere](https://www.pythonanywhere.com)

## About The Project
This project is provided by my [Instructur](https://github.com/imamcs19/FGA-Big-Data-Using-Python-Filkom-x-Mipa-UB-2021) with some little edit from me. This project supposed to just provide ready enviromental web development and modified by you, not supposed to be final product.

![feature](https://github.com/f3rry12/EasyDeployFlask/blob/main/readMeAsset/provide1.jpg)

Feature (by default / before you edit it) :
* Login and Register
* Machine Learning with Linear Regression using SKLearn and PySpark and visualize the result in graph.
* Upload File, Download File, and File Management
* API


## Getting Started
I made this tutorial intended for beginner

### Prerequisites

What you need to prepare:
* Have verified account at https://www.pythonanywhere.com
* Install Spark (opsional, its required if you use ML that need Spark). You can follow my guide in my other repository  [install-pyspark-split1](https://github.com/f3rry12/install-pyspark-split1) (you can install it later after deploy this web project)

### How to Deploy

1. Open [pythonanywhere](https://www.pythonanywhere.com) (make sure to login), go to web section , then click add a new web app
![makenewweb](https://github.com/f3rry12/EasyDeployFlask/blob/main/readMeAsset/ss1.jpg)

  Just click next <br>
![next](https://github.com/f3rry12/EasyDeployFlask/blob/main/readMeAsset/ss2.jpg)

  Choose Flask <br>
![chooseflask](https://github.com/f3rry12/EasyDeployFlask/blob/main/readMeAsset/ss3.jpg)

  You can choose whatever version, but I recommend to use 3.7 if you don’t have plan in mind
![choose37](https://github.com/f3rry12/EasyDeployFlask/blob/main/readMeAsset/ss3_5.jpg)

  Just click Next <br>
![next](https://github.com/f3rry12/EasyDeployFlask/blob/main/readMeAsset/ss4.jpg)

Now your site server in [pythonanywhere](https://www.pythonanywhere.com) ready.

2. Go to Files section, then create new directory called tar
![createtar](https://github.com/f3rry12/EasyDeployFlask/blob/main/readMeAsset/ss5.jpg)

3. After you make tar directory, back to main directory then choose mysite directory
   Then delete flask_app.py (we don’t need it because we already have this file ready in this git repository)
![deleteflaskapp](https://github.com/f3rry12/EasyDeployFlask/blob/main/readMeAsset/ss6.jpg)

4. Go to Consoles section, then click Bash
![openbash](https://github.com/f3rry12/EasyDeployFlask/blob/main/readMeAsset/ss7.jpg)

Then follow this command to clone this repository
   ```sh
   cd tar
   ```
   ```sh
   git clone https://github.com/f3rry12/EasyDeployFlask.git
   ```
![clonerepo](https://github.com/f3rry12/EasyDeployFlask/blob/main/readMeAsset/ss8.jpg)

Move the project from tar directory to mysite directory
   ```sh
   cd
   ```
   ```sh
   mv /home/yourusername/tar/EasyDeployFlask/* /home/yourusername/mysite/
   ```
![moveclone](https://github.com/f3rry12/EasyDeployFlask/blob/main/readMeAsset/ss9.jpg)


5. Install module flask cors 
   ```sh
   pip3.7 install --user flask_cors
   ```
![intallflaskcors](https://github.com/f3rry12/EasyDeployFlask/blob/main/readMeAsset/ss10.jpg)

6. Go to Web section, then click reload button
![reloadwev](https://github.com/f3rry12/EasyDeployFlask/blob/main/readMeAsset/ss11.jpg)

You can visit your web to check if it success
![visitweb](https://github.com/f3rry12/EasyDeployFlask/blob/main/readMeAsset/ss12.jpg)

7. Last, because free user [pythonanywhere](https://www.pythonanywhere.com) have limited space. Try clear some space by deleting tar directory and read me asset

## What to edit?
In this guide, I can not give you detailed tutorial what to edit and how to do it. Instead, i just give you some mapping about this project
![mapping](https://github.com/f3rry12/EasyDeployFlask/blob/main/readMeAsset/edit1.png)
Mostly you will edit flask_app.py and some HTML files

To edit this project, you need some knowledge about :
* Python programing (obviously)
* Flask infrastructure
* Basic HTML
* SQLite (You dont need to study from scracth, just googling what you need to edit)


## Acknowledgements
Special thanks to my Instructur [Imam Cholissodin](https://github.com/imamcs19), S.Si., M.Kom. for providing this project <br>
Asisten Instructur Yonas Asmara <br>
Collaboration from Fakultas Ilmu Komputer (Filkom) x MIPA, Brawijaya University (UB) 2021
