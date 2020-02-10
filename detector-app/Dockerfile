FROM python:3.8-buster as darknet-builder
WORKDIR /darknet
RUN apt-get update -y && apt-get install -y git sed make gcc g++ libc-dev libgomp1
RUN git clone https://github.com/pjreddie/darknet /darknet
RUN sed -i "/OPENMP=0/c\\OPENMP=1" Makefile
RUN cd /darknet && make clean && make

FROM python:3.8-buster as app
WORKDIR /app
EXPOSE 8000
RUN pip install falcon gunicorn Pillow
COPY --from=darknet-builder /darknet/libdarknet.so .
COPY . .
ENTRYPOINT gunicorn -b 0.0.0.0:8000 app:app