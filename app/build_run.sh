rm -rf webapp/*
cd ../tfjs-models/pose-detection/demos/upload_video/
yarn build
cd ../../../../app/
docker build --platform linux/amd64 -t aitenniscoach:latest .
docker image save -o ../aitenniscoach_image.tar aitenniscoach:latest
docker run -d -p 3000:3000 aitenniscoach:latest
