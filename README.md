cloud-python-opencv
===================

A small set of startup scripts to install everything we need on amazon EC2.
The main script is `boostrap` that installs:
 * **python 2.7** (instead of 2.6)
 * **numpy 1.9**
 * **opencv 2.4.9** with `ffmpeg` support
 * **Dropbox**
 * [py-video-retargeting][1] repository
 * [py-seam-merging][2] repository
 * [py-tvd][3] repository

[1]: https://github.com/PNProductions/py-video-retargeting
[2]: https://github.com/PNProductions/py-seam-merging
[3]: https://github.com/PNProductions/py-tvd

## Installation
 After lanching an Amazon EC2 instance:
```shell
ssh using the certificate given _ssh -i CERTIFICATE.pem ec2-user@IP_ADDRESS_
#change to amazon shell
sudo yum update
sudo yum install -y git
git clone https://github.com/PNProductions/cloud-python-opencv
./bootstrap
```
To check if everything is installed properly run
```shell
./test
```
You'll get a version of each package installed, for example:
```shell
Python version --> 2.7.5
Numpy version --> 1.9.1
Opencv version --> 2.4.9
Cython version --> Version: 0.21.1
Image seam merging version --> Version: 0.2.4
Video seam merging version --> Version: 0.1
TVD version --> Version: 1.0
Dropbox is running correctly
```
##Example usage
To start the algorithm run
```shell
python video_retargeting.py
```
You can also use the following options:
* `-s SEAM`: specify the seam to use (default is 1)
* `-f FRAME`: specify the frame number to consider (default is 10)
* `-p PATH`: specify the path of the input videos
* `-i`: save the importance map
* `-nv`: do not save motion vector
* `-g`: use motion vector for the whole video and not frame by frame
* `-m s{seam_merging,seam_carving,time_merging}`:
  * `seam_merging`: use seam merging algorithm (default)
  * `seam_carving`: use Rubinstein seam carving method with forward energy
  * `time_merging`: apply a temporally resize instead that a width resize. A seam is a frame, so, if you want to delete 10 frame from the video use the option `-s 10`
