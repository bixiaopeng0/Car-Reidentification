import cv2
#黑
img_path = 'G:/19\lab\python\Vehicle Re-identification\doc\dataset\VRID\image/1\License_6/4404000000002943547902.jpg'
#金色
# img_path = 'G:/19\lab\python\Vehicle Re-identification\doc\dataset\VRID\image/4\License_302/4404000000002947607742.jpg'
#白色
# img_path = 'G:/19\lab\python\Vehicle Re-identification\doc\dataset\VRID\image/6\License_522/4404000000002940377174.jpg'
#红色
# img_path = 'G:/19\lab\python\Vehicle Re-identification\doc\dataset\VRID\image/2\License_109/4404000000002946026672.jpg'
#绿色
# img_path = 'G:/19\lab\python\Vehicle Re-identification\doc\dataset\VRID\image/3\License_203/4404000000002940410681.jpg'
img = cv2.imread(img_path)
def find_car_contour(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # dst = cv2.Canny(gray, 100, 200)
    x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(gray, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)  # 转回uint8 
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    ret, binary = cv2.threshold(dst, 20, 255, cv2.THRESH_BINARY)
    # cv2.imshow("sobel", binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.dilate(binary,kernel)
    binary = cv2.erode(binary,kernel)
    # binary = cv2.dilate(binary,kernel)
    binary = cv2.dilate(binary, kernel)
    # cv2.imshow("thresh_binary", binary)
    contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cv2.waitKey()
    for i in range(0, len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        # rect = cv2.minAreaRect(contours[i])
        # width,height  = rect[1]
        area = img.shape[0] * img.shape[1] #shape[0]--height
        if w*h > area/2:
            cv2.rectangle(img, (x, y+int(h/2)), (x + w, y + h), (153, 153, 233), 5)
            # cv2.rectangle(img, rect[0],rect[1], (153, 153, 0), 5)
            # cv2.drawContours(img,contours[i],-1, (153, 153, 0), 5)
            # cv2.imshow("img",img)
            # cv2.waitKey()
            return (x, y, w, h)
            # print(rect)

    return 0,0,img.shape[1],img.shape[0]
if __name__ == '__main__':
    # pass
  # print(cv2.useOptimized())
  find_car_contour(img)
  cv2.imshow("img",img)
  cv2.waitKey()
