#!/sh

export TMPDIR=/var/tmp
sudo yum update -y
sudo yum install python311 python3-devel gcc postgresql-devel git -y
curl -O https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py --user
pip3 install --upgrade pip
pip3 install python-dotenv pypdf pdfminer.six trulens_eval psycopg2 psutil IPython langchain langchain-community llama-index astrapy
