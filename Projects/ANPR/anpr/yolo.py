import colorsys
import os
import cfg
import cv2
import operator
import pytesseract
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
from yolov3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolov3.utils import letterbox_image
from keras.utils import multi_gpu_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from label import point_inside_of_quad, nms, resize_image
from network import East

os.environ['KERAS_BACKEND'] = 'tensorflow'
pytesseract.pytesseract.tesseract_cmd = r'tesseract\\tesseract.exe'


class YOLO:
    _defaults = {
        "model_path": 'resources/trained_weights_stage_1.h5',
        "anchors_path": 'resources/custom_anchors.txt',
        "classes_path": 'resources/plate_class.txt',
        "score": 0.3,
        "iou": 0.2,
        "model_image_size": (640, 640),
        "gpu_num": 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self):
        K.clear_session()
        self.__dict__.update(self._defaults)  # set up default values
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors == 6  # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
            self.yolo_model.load_weights(self.model_path)  # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2,))
        if self.gpu_num >= 2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        font = ImageFont.truetype(font='resources/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        cfg.clipboard = ''
        K.clear_session()
        global east_detect
        global east
        east = East()
        east_detect = east.east_network()
        east_detect.load_weights(cfg.saved_model_weights_file_path)

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            ###
            img = image.copy()
            if top != bottom and left != right:
                cropped_img = img.crop((left, top, right, bottom))
                cropped_img = np.array(cropped_img)
                cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite('resources\\cropped.jpg', cropped_img)
                str_result = recognize()
                cfg.clipboard = cfg.clipboard + str_result + ' '

                # drawing
                for j in range(thickness):
                    draw.rectangle(
                        [left + j, top + j, right - j, bottom - j],
                        outline=self.colors[c])
                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=self.colors[c])
                draw.text(text_origin, str_result, fill=(0, 0, 0), font=font)

        result = np.asarray(image)
        return result

    def close_session(self):
        self.sess.close()


def image_to_text_tesseract(warped_img):
    warped_img = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    _, warped_img = cv2.threshold(warped_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    blue = [255]
    warped_img = cv2.copyMakeBorder(warped_img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=blue)
    result_str = pytesseract.image_to_string(warped_img, lang='eng', config='--psm 7')
    result_str = result_str.translate(str.maketrans('£sSmcky¥Yxa', 'E55MCKУУУXA', ' .,\'`>-'))
    return result_str


def create_blank(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)
    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color
    return image


def fix_common_mistakes(some_str):
    if len(some_str) == 6:
        changed_str = list(some_str)

        if changed_str[0] == '0' or changed_str[0] == 'o' or changed_str[0] == 'Q':
            changed_str[0] = 'O'
        if changed_str[0] == '8':
            changed_str[0] = 'B'

        if changed_str[1] == 'o' or changed_str[1] == 'O' or changed_str[1] == 'Q':
            changed_str[1] = '0'
        if changed_str[1] == 's' or changed_str[1] == 'S':
            changed_str[1] = '5'
        if changed_str[1] == 'I' or changed_str[1] == 'i':
            changed_str[1] = '1'
        if changed_str[1] == 'B':
            changed_str[1] = '8'

        if changed_str[2] == 'o' or changed_str[2] == 'O' or changed_str[2] == 'Q':
            changed_str[2] = '0'
        if changed_str[2] == 's' or changed_str[2] == 'S':
            changed_str[2] = '5'
        if changed_str[2] == 'I' or changed_str[2] == 'i':
            changed_str[2] = '1'
        if changed_str[2] == 'B':
            changed_str[2] = '8'

        if changed_str[3] == 'o' or changed_str[3] == 'O' or changed_str[3] == 'Q':
            changed_str[3] = '0'
        if changed_str[3] == 's' or changed_str[3] == 'S':
            changed_str[3] = '5'
        if changed_str[3] == 'I' or changed_str[3] == 'i':
            changed_str[3] = '1'
        if changed_str[3] == 'B':
            changed_str[3] = '8'

        if changed_str[4] == '0' or changed_str[4] == 'o' or changed_str[4] == 'Q':
            changed_str[4] = 'O'
        if changed_str[4] == '8':
            changed_str[4] = 'B'

        if changed_str[5] == '0' or changed_str[5] == 'o' or changed_str[5] == 'Q':
            changed_str[5] = 'O'
        if changed_str[5] == '8':
            changed_str[5] = 'B'

        final_str = "".join(changed_str)
    else:
        final_str = some_str

    return final_str


def recognize():
    path_to_img = 'resources\\cropped.jpg'
    cropped_img = cv2.imread('resources\\cropped.jpg')
    height = np.size(cropped_img, 0)
    width = np.size(cropped_img, 1)
    nh = 256 / height
    nw = 256 / width

    img = image.load_img(path_to_img)
    img = img.resize((256, 256), Image.NEAREST).convert('RGB')
    path_to_img = 'resources\\1.jpg'
    img.save(path_to_img)

    status_ret, pts1, pts2 = predict()
    img = cv2.imread('resources\\1.jpg')

    if status_ret == 2:
        pts1[0][0] /= nw
        pts1[0][1] /= nh
        pts1[1][0] /= nw
        pts1[1][1] /= nh
        pts1[2][0] /= nw
        pts1[2][1] /= nh
        pts1[3][0] /= nw
        pts1[3][1] /= nh

        pts2[0][0] /= nw
        pts2[0][1] /= nh
        pts2[1][0] /= nw
        pts2[1][1] /= nh
        pts2[2][0] /= nw
        pts2[2][1] /= nh
        pts2[3][0] /= nw
        pts2[3][1] /= nh

        warped1 = four_point_transform(cropped_img, pts1)
        string1 = image_to_text_tesseract(warped1)

        warped2 = four_point_transform(cropped_img, pts2)
        string2 = image_to_text_tesseract(warped2)

        if len(string1) > len(string2):
            string1 = fix_common_mistakes(string1)
            final_str = string1 + ' ' + string2
        else:
            string2 = fix_common_mistakes(string2)
            final_str = string2 + ' ' + string1

        return final_str

    if status_ret == 1:  # TODO: Raise error
        return 'only one box..'

    return 'no one..'  # TODO: Raise error


def sigmoid(x):
    """`y = 1 / (1 + exp(-x))`"""
    return 1 / (1 + np.exp(-x))


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


def cut_text_line(geo, scale_ratio_w, scale_ratio_h, im_array, img_path, s):
    geo /= [scale_ratio_w, scale_ratio_h]
    p_min = np.amin(geo, axis=0)
    p_max = np.amax(geo, axis=0)
    min_xy = p_min.astype(int)
    max_xy = p_max.astype(int) + 2
    sub_im_arr = im_array[min_xy[1]:max_xy[1], min_xy[0]:max_xy[0], :].copy()
    for m in range(min_xy[1], max_xy[1]):
        for n in range(min_xy[0], max_xy[0]):
            if not point_inside_of_quad(n, m, geo, p_min, p_max):
                sub_im_arr[m - min_xy[1], n - min_xy[0], :] = 255
    sub_im = image.array_to_img(sub_im_arr, scale=False)
    sub_im.save(img_path + '_subim%d.jpg' % s)


def PolyArea(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def predict():
    img_path = 'resources\\1.jpg'
    img = image.load_img(img_path)
    pixel_threshold = cfg.pixel_threshold

    d_wight, d_height = resize_image(img, cfg.max_predict_img_size)
    img = img.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
    img = image.img_to_array(img)
    img = preprocess_input(img, mode='tf')
    x = np.expand_dims(img, axis=0)
    y = east_detect.predict(x)

    y = np.squeeze(y, axis=0)
    y[:, :, :3] = sigmoid(y[:, :, :3])
    cond = np.greater_equal(y[:, :, 0], pixel_threshold)
    activation_pixels = np.where(cond)
    quad_scores, quad_after_nms = nms(y, activation_pixels)

    with Image.open(img_path) as im:
        im_array = image.img_to_array(im.convert('RGB'))
        d_wight, d_height = resize_image(im, cfg.max_predict_img_size)
        scale_ratio_w = d_wight / im.width
        scale_ratio_h = d_height / im.height
        im = im.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
        draw = ImageDraw.Draw(im)
        for i, j in zip(activation_pixels[0], activation_pixels[1]):
            px = (j + 0.5) * cfg.pixel_size
            py = (i + 0.5) * cfg.pixel_size
            line_width, line_color = 1, 'red'
            if y[i, j, 1] >= cfg.side_vertex_pixel_threshold:
                if y[i, j, 2] < cfg.trunc_threshold:
                    line_width, line_color = 2, 'yellow'
                elif y[i, j, 2] >= 1 - cfg.trunc_threshold:
                    line_width, line_color = 2, 'green'
            draw.line([(px - 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size),
                       (px + 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size),
                       (px + 0.5 * cfg.pixel_size, py + 0.5 * cfg.pixel_size),
                       (px - 0.5 * cfg.pixel_size, py + 0.5 * cfg.pixel_size),
                       (px - 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size)],
                      width=line_width, fill=line_color)

        txt_items = []
        for score, geo, s in zip(quad_scores, quad_after_nms,
                                 range(len(quad_scores))):
            if cfg.predict_cut_text_line:
                cut_text_line(geo, scale_ratio_w, scale_ratio_h, im_array,
                              img_path, s)
            rescaled_geo = geo / [scale_ratio_w, scale_ratio_h]
            rescaled_geo_list = np.reshape(rescaled_geo, (8,)).tolist()
            txt_item = ','.join(map(str, rescaled_geo_list))
            txt_items.append(txt_item + '\n')

        if len(txt_items) > 0:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            check = True
            while check:
                check = False
                for i in range(0, len(txt_items)):
                    if i < len(txt_items):
                        some_str = txt_items[i].replace('\n', '')
                        some_str = some_str.split(',')
                        for z in range(0, 8):
                            if float(some_str[z]) < 0:
                                check = True
                        if check == True:
                            check = False
                            txt_items.pop(i)

            if len(txt_items) > 2:  # >2 bounding boxes
                area_dict = {}
                for i in range(0, len(txt_items)):
                    if i < len(txt_items):
                        some_str = txt_items[i].replace('\n', '')
                        some_str = some_str.split(',')
                        x = [int(float(some_str[0])), int(float(some_str[2])), int(float(some_str[4])),
                             int(float(some_str[6]))]
                        y = [int(float(some_str[1])), int(float(some_str[3])), int(float(some_str[5])),
                             int(float(some_str[7]))]
                        area_dict[i] = PolyArea(x, y)

                Max1 = max(area_dict.items(), key=operator.itemgetter(1))[0]
                del area_dict[Max1]

                Max2 = max(area_dict.items(), key=operator.itemgetter(1))[0]
                del area_dict[Max2]

                str_xy = txt_items[Max1].replace('\n', '')
                str_xy = str_xy.split(',')
                pts1 = get_pts(str_xy)

                str_xy = txt_items[Max2].replace('\n', '')
                str_xy = str_xy.split(',')
                pts2 = get_pts(str_xy)
                return (2, pts1, pts2)

            elif len(txt_items) == 2:  # 2 bounding boxes
                str_xy = txt_items[0].replace('\n', '')
                str_xy = str_xy.split(',')
                pts1 = get_pts(str_xy)

                str_xy = txt_items[1].replace('\n', '')
                str_xy = str_xy.split(',')
                pts2 = get_pts(str_xy)
                return (2, pts1, pts2)

            elif len(txt_items) == 1:  # 1 bounding box
                str_xy = txt_items[0].replace('\n', '')
                str_xy = str_xy.split(',')
                pts1 = get_pts(str_xy)
                pts2 = []
                return (1, pts1, pts2)

            elif len(txt_items) <= 0:  # no bounding boxes
                pts1 = []
                pts2 = []
                return (0, pts1, pts2)


def get_pts(str_xy):
    x1, y1, x2, y2, x3, y3, x4, y4 = str_xy[0], str_xy[1], str_xy[2], str_xy[3], \
                                     str_xy[4], str_xy[5], str_xy[6], str_xy[7]
    coord = np.array([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])
    pts = np.array(coord, dtype="float32")
    return pts


def run_detection():
    if cfg.str_file:
        try:
            image = Image.open(cfg.str_file)
        except:
            print('File ' + cfg.str_file + ' Open Error! ')
        else:
            yolo = YOLO()
            r_image = yolo.detect_image(image)
            r_image = cv2.cvtColor(r_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite('resources\\TEMP\\detected.jpg', r_image)
            yolo.close_session()
