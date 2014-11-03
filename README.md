cloud-python-opencv
===================

A small set of startup scripts to install everything we need on amazon EC2.
The main script is `boostrap` that installs:
 * **python 2.7** (instead that 2.6)
 * **numpy 1.9**
 * **opencv 2.4.9** with `ffmpeg` support
 * **Dropbox**
 * [py-video-retargeting][1] repository
 * [py-seam-merging][2] repository

[1]: https://github.com/PNProductions/py-video-retargeting
[2]: https://github.com/PNProductions/py-seam-merging

## Example usage
 After lanching an Amazon EC2 instance:
```shell
ssh using the certificate given _ssh -i CERTIFICATE.pem ec2-user@IP_ADDRESS_
#change to amazon shell
sudo yum update
sudo yum install -y git
git clone https://github.com/PNProductions/cloud-python-opencv
./bootstrap
```
