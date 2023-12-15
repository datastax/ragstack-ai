#!/sh

export TMPDIR=/var/tmp
sudo yum update -y
sudo yum install python311 python3-pip -y
pip3 install --upgrade pip
pip3 install ragstack-ai ragas==0.0.19 python-dotenv pypdf pdfminer.six
