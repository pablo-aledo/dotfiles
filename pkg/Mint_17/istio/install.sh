cd
curl -L https://github.com/istio/istio/releases/download/0.7.1/istio-0.7.1-linux.tar.gz | tar xz
cd istio-0.7.1; export PATH=$PWD/bin:$PATH
kubectl apply -f install/kubernetes/istio.yaml
