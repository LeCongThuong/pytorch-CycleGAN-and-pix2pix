apt-get install git
git clone -b cyclegan https://github.com/LeCongThuong/pytorch-CycleGAN-and-pix2pix.git
pip install virtualenv
virtualenv -p python3 .env
source .env/bin/activate
pip install dvc
pip install dvc[gdrive]
cd /content/pytorch-CycleGAN-and-pix2pix.git
dvc pull

