import cv2
def translate(image, x, y):
    '''
    translation matrix. - koliko se pixela mice slika ljevo-desno, gore-dole
    #[1,0,tx] tx-koliko micemo sliku ljevo ili desno 
    #(pozitivno desno, negativno ljevo)
    #[0,1,ty] ty- koliko micemo sliku gore ili dole
    #(pozitivno dolje, negativno gore)
    args:
    image - slika koju zelimo translatirati
    x   - broj pixela (-ljevo, +desno)
    y   - broj bixela (-gore, +dolje)
    '''
    M = np.float32([[1,0,x], [0,1,y]]) 
    shifted = cv2.warpAffine(image, M,(image.shape[1],image.shape[0]))
    return shifted


def rotate(image, angle, center=None, scale=1.0):
    '''
    rotation matrix - rotacija slike oko osi(default=centar slike) za x stupnjeva
    args:
    image - slika koju zelimo rotirati
    angle - kut(stupnjevi) za koliko zelimo rotirati sliku
    center - tocka iz koje rotiramo(default=center)
    scale  - razmjer po kojem uvecavamo ili smanjujemo sliku (default 1)
    '''
    #određivanje velicine slike
    (h,w) = image.shape[:2]
   
    #u slicaju da nije navedena tocka roatacije onda definiramo centar slike(center=None)
    if center is None:
         center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w,h))
    return rotated
    
def resize(image, width=None, height=None, inter = cv2.INTER_AREA):
    '''
    resize image funkcija - pomocu sirine ili visine slike
    
    args:
    image - slika koju zelimo resizat
    height - (default None) nova visina slike(odabrati jednu ili drugu)
    width - (default None) nova sirina slike(odabrati jednu ili drugu)
    inter - cv2 metoda interpolacije za resize default(cv2.INTER_AREA)
    
    '''
    
    dim = None
    # velicina orginalne slike u pixelima
    (h, w) = image.shape[:2]
    
    # test ako nije navedena sirina i visina vraca izvornu sliku
    if width is None and height is None:
        return image
    
    # ako koristimo sirinu za resize
    # racunanje omjera r pomocu nove visine / orginal visine(float zbog omjera)
    # racunanje nove dimenzije (dim) stara sirina puta omjer, nova visina
    if width is None:
        r = height / float(h)
        dim = (int(w*r), height)
    
    # ako koristimo visinu za resize
    # racunanje omjera r pomocu nove sirine / orginal sirine(float zbog omjera)
    # racunanje nove dimenzije (dim): nova sirina, stara visina puta omjer
    else:
        r = width / float(w)
        dim = (width, int(h*r))
    
    resized = cv2.resize(image, dim, interpolation=inter)
    
    return resized
    


def open_and_prepare():
    return

def find_sections():
    return

def build_dir():
    return

def save_and_copy():
    return
