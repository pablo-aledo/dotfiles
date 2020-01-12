docker pull milvusdb/milvus:0.6.0-cpu-d120719-2b40dd

# Create Milvus file
mkdir -p /home/$USER/milvus/conf
cd /home/$USER/milvus/conf
wget https://raw.githubusercontent.com/milvus-io/milvus/master/core/conf/demo/server_config.yaml
wget https://raw.githubusercontent.com/milvus-io/milvus/master/core/conf/demo/log_config.conf

# Start Milvus
docker run -d --name milvus_cpu \
-p 19530:19530 \
-p 8080:8080 \
-v /home/$USER/milvus/db:/var/lib/milvus/db \
-v /home/$USER/milvus/conf:/var/lib/milvus/conf \
-v /home/$USER/milvus/logs:/var/lib/milvus/logs \
milvusdb/milvus:0.6.0-cpu-d120719-2b40dd

# Install Milvus Python SDK
pip3 install pymilvus==0.2.6

# Download Python example
wget https://raw.githubusercontent.com/milvus-io/pymilvus/blob/0.6.0/examples/example.py
wget https://raw.githubusercontent.com/milvus-io/pymilvus/master/examples/example.py

# Run Milvus Python example
python3 example.py
