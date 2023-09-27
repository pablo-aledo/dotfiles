package main

import (
	"log"
	"net/http"
	_ "net/http/pprof"
)

func main() {
	http.HandleFunc("/", handle)
	log.Fatal(http.ListenAndServe(":8080", nil))
}

func handle(w http.ResponseWriter, r *http.Request) {
	log.Printf("handling request from: %s", r.RemoteAddr)
	if _, err := w.Write([]byte(r.RemoteAddr)); err != nil {
		log.Printf("could not write IP: %s", err)
	}
}
