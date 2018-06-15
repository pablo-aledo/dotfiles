cd
curl -L https://github.com/istio/istio/releases/download/0.7.1/istio-0.7.1-linux.tar.gz | tar xz
cd istio-0.7.1; export PATH=$PWD/bin:$PATH
kubectl apply -f install/kubernetes/istio.yaml

./install/kubernetes/webhook-create-signed-cert.sh --service istio-sidecar-injector --namespace istio-system --secret sidecar-injector-certs
kubectl apply -f  install/kubernetes/istio-sidecar-injector-configmap-release.yaml
cat install/kubernetes/istio-sidecar-injector.yaml | ./install/kubernetes/webhook-patch-ca-bundle.sh > install/kubernetes/istio-sidecar-injector-with-ca-bundle.yaml
kubectl apply -f install/kubernetes/istio-sidecar-injector-with-ca-bundle.yaml

openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout /tmp/tls.key -out /tmp/tls.crt -subj "/CN=foo.bar.com"
kubectl create -n istio-system secret tls istio-ingress-certs --key /tmp/tls.key --cert /tmp/tls.crt
kubectl create namespace istio-ns

