# HW4 ― Videos
In this assignment, you will learn to perform both trimmed action recognition and temporal action segmentation in full-length videos.

<p align="center">
  <img width="750" height="250" src="https://lh3.googleusercontent.com/j48uA36UbZp3KR41opZUzntxhlJWoX_R5joeNsTGMN2_cSXI0UFNKuKVu8em_txzOIVbnU8p_oOb">
</p>



For this dataset, the action labels are defined as below:

|       Action      | Label |
|:-----------------:|:-----:|
| Other             | 0     |
| Inspect/Read      | 1     |
| Open              | 2     |
| Take              | 3     |
| Cut               | 4     |
| Put               | 5     |
| Close             | 6     |
| Move Around       | 7     |
| Divide/Pull Apart | 8     |
| Pour              | 9     |
| Transfer          | 10    |

⚠️ Important Note⚠️
In this homework, you are not allowed to use any external datset except for the ImageNet dataset (You can load any model pre-trained only on the ImageNet dataset)

### Packages
This homework should be done using python3.6 and you can use all the python3.6 standard libraries. For a list of third-party packages allowed to be used in this assignment, please refer to the requirments.txt for more details.
You can run the following command to install all the packages listed in the requirements.txt:

    pip3 install -r requirements.txt

Note that using packages with different versions will very likely lead to compatibility issues, so make sure that you install the correct version if one is specified above. E-mail or ask the TAs first if you want to import other packages.

