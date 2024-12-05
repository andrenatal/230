FROM node:22.12.0-slim
ADD . /app
WORKDIR /app
EXPOSE 3000

ENTRYPOINT ["npx", "http-server", "-s", ".", "-p", "3000", "--cors", "watch"]