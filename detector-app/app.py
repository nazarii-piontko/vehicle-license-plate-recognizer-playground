import falcon
from PIL import Image, ImageDraw

import darknet as dn

net = dn.load_net(bytes('yolo.cfg', encoding='utf-8'), bytes('yolo.backup', encoding='utf-8'), 0)


def detect():
    print('Loading image')

    # Use tmp.jpg for demo purposes
    im = dn.load_image(bytes('tmp.jpg', encoding='utf-8'), 0, 0)
    num = dn.c_int(0)
    pnum = dn.pointer(num)
    
    print('Predicting image')
    
    dn.predict_image(net, im)
    
    print('Getting boxes')
    
    dets = dn.get_network_boxes(net, im.w, im.h, 0.5, 0.5, None, 1, pnum)

    print('Marking boxes')

    res = []
    classes = 1
    for j in range(num.value):
        for i in range(classes):
            if dets[j].prob[i] > 0.75:
                b = dets[j].bbox
                res.append((b.x, b.y, b.w, b.h))
    dn.free_image(im)
    dn.free_detections(dets, num)

    print('Saving image')

    source_img = Image.open('tmp.jpg').convert("RGB")
    size = source_img.size
    w = size[0]
    h = size[1]

    draw = ImageDraw.Draw(source_img)

    for b in res:
        x1 = (b[0] - b[2] / 2.) * w
        x2 = (b[0] + b[2] / 2.) * w
        y1 = (b[1] - b[3] / 2.) * h
        y2 = (b[1] + b[3] / 2.) * h
        draw.rectangle(((x1, y1), (x2, y2)), outline="red")
        print(b)

    source_img.save('tmp.jpg', "JPEG")


class DetectorResource(object):
    def on_get(self, req, resp):
        resp.status = falcon.HTTP_200
        resp.content_type = 'text/html'
        with open('index.html', 'r') as f:
            resp.body = f.read()

    def on_post(self, req, resp):
        raw = req.bounded_stream.read()

        # Use tmp.jpg for demo purposes
        with open('tmp.jpg', 'wb') as output_file:
            output_file.write(raw)

        detect()

        resp.status = falcon.HTTP_200
        resp.content_type = 'image/jpeg'
        with open('tmp.jpg', 'rb') as f:
            resp.body = f.read()


app = falcon.API()
res = DetectorResource()
app.add_route('/detector', res)
