FROM node:22.12.0-slim
RUN apt-get update -y; apt-get install -y python3 python3-pip
ADD tfjs-models /tfjs-models
WORKDIR /tfjs-models/pose-detection/demos/upload_video
RUN rm -rf .cache dist node_modules
RUN yarn build-dep
RUN yarn
EXPOSE 1234

ENTRYPOINT ["yarn", "watch"]