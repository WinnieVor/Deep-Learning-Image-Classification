#!/usr/bin/env python
# coding: utf-8

# Winnie VORIHILALA <br />
# MS ESD 2019-2020 <br />
# INSA Rouen <br />

# # <center> TP Deep Learning : Classification d'images </center>
# 

# L'obectif de ce TP consiste à catégoriser des images en 2 classes : chihuahua et muffin puis à prédire si de nouvelles images de test appartiennent à la classe "chihuahua" ou à la classe "muffin". Pour cela, nous allons tester 3 méthodes :
# - la première consiste à entrainer notre propre modèle puis à effectuer notre prédiction à partir de ce modèle entrainé
# - la 2ème consiste à utiliser la méthode bottleneck information à partir d'un modèle déjà entrainé
# - la 3ème consiste à faire du fine-tuning à partir d'un modèle déjà entrainé.
# <br />
#  
# Nous allons comparer les résultats obtenus par ces 3 méthodes et déterminer laquelle permet d'obtenir le meilleur score.

# In[4]:


#Connexion à mon drive où sont stockés les dataset
from google.colab import drive
drive.mount("/content/gdrive/")


# Afin que l'environnement d'execution soit bien GPU, aller dans Execution/Modifier le type d'exécution/Selectionner GPU dans Accélérateur matériel.
# Il est nécessaire d'effectuer cette manipulation car certaines méthodes qui seront vues dans ce TP, notamment le fine tuning nécessite beaucoup de puissance de calcul

# In[5]:


#Verification de l'environnement d'execution
import tensorflow as tf
tf.test.gpu_device_name()


# In[6]:


#Affichage des caractéristiques de l'environnement d'execution
from tensorflow.python.client import device_lib
device_lib.list_local_devices()


# In[7]:


#Vérification de la version de tensorflow
get_ipython().run_line_magic('tensorflow_version', '')
#%tensorflow_version 1.x : pour selectionner la v1 de tensorflow


# In[8]:


get_ipython().system('nvidia-smi')


# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img

import matplotlib.pyplot as plt


# In[ ]:


# Chargement des dataset depuis mon drive
train_data_dir = 'gdrive/My Drive/Data/chihuahua-vs-muffin/train'
validation_data_dir = 'gdrive/My Drive/Data/chihuahua-vs-muffin/validation'
test_data_dir = 'gdrive/My Drive/Data/chihuahua-vs-muffin/test'
#print(type(train_data_dir))


# # 1ère méthode : convolutional neural network with original dataset

# In[13]:


#Test de lecture d'une image quelconque pour vérifier que nos dataset ont bien été chargés 
im = plt.imread('gdrive/My Drive/Data/chihuahua-vs-muffin/train/chihuahua/ActiOn_1.jpg')

plt.figure(figsize=(15,9))
plt.imshow(im)
plt.axis('off')
plt.tight_layout()


# In[ ]:


#Initialisation du nombre d'itérations (epochs) et du nombre de lots par itération (batch_size)
epochs = 50 #initialement à 5, attention ne pas prendre trop petit car ralentit le temps d'execution 
batch_size = 32 #initialement à 16, attention ne pas prendre trop petit car ralentit le temps d'execution


# In[94]:


#Configuration de l'augmentation des données que l'on utilisera lors de notre apprentissage

#Ce procédé est utile lorsqu'on a peu de données d'apprentissage.
#ImageDataGenerator permet d'augmenter en temps réel les données d'apprentissage en générant des images artificielles 
#chacune différente les unes des autres grace au réglages des différentes paramètres de la fonction

train_datagen = ImageDataGenerator( 
    rescale=1. / 255, #la division par 255 permet de normaliser les photos
    #shear_range=0.2, #intensité de cisaillement (angle de cisaillement dans le sens antihoraire en degrés)
    shear_range=50,
    zoom_range=0.2, #plage de zoom aléatoire
    horizontal_flip=True, #retourne aléatoirement les entrées verticalement
    #vertical_flip=True #retourne aléatoirement les entrées verticalement
    )

# this is the augmentation configuration we will use for testing:
# only rescaling
valid_datagen = ImageDataGenerator(rescale=1. / 255)

test_datagen = ImageDataGenerator(rescale=1. / 255)

# dimensions of our images.
img_width, img_height = 256,256

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = valid_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')


# Nos données d'apprentissage se composent de 568 images appartenant aux 2 classes chihuahua et muffin. <br>
# Nos données de validation se composent de 141 images appartenant aux 2 classes chihuahua et muffin. <br>

# In[ ]:


image_batch, label_batch = next(train_generator)


# In[ ]:


#Fonction qui permet la visualisation d'un extrait de notre training set incluant les images générées artificiellement avec ImageGenerator()
CLASS_NAMES = list(train_generator.class_indices.keys())
def show_batch(image_batch, label_batch):
  plt.figure(figsize=(10,10))
  for n in range(16):
      ax = plt.subplot(4,4,n+1)
      plt.imshow(image_batch[n])
      plt.title(CLASS_NAMES[label_batch[n].argmax()].title())
      plt.axis('off')


# In[97]:


show_batch(image_batch, label_batch)


# Les images distordues ci-dessus sont les images qui ont été générées artificiellement avec la fonction ImageGenerator.

# In[ ]:


nb_train_samples = train_generator.n
nb_validation_samples = validation_generator.n
nb_test_samples = test_generator.n


# In[ ]:


#Creation du modèle de réseau de neurones 

input_shape = (img_height, img_width, 3) #3 car les images sont en couleur
model = Sequential() 
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten()) #permet de mettre les images bout à bout
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.1)) #10% des neurones seront ignorées

model.add(Dense(2)) #2 car nos avons 2 classes : chihuahua et muffin
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='Adam', 
              metrics=['accuracy'])


# In[100]:


model.summary()


# In[102]:


model.fit(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    verbose=2)


# In[103]:


model.evaluate(train_generator)


# In[105]:


model.evaluate(validation_generator)


# In[107]:


model.evaluate(test_generator)


# Cette méthode nous donne une accuracy score de :
# - 99,29% sur les données d'apprentissage
# - 91,49% sur les données de validation
# - 62,50% sur les données de test.
# 
# Nous obtenons des taux satisfaisants dans l'ensemble sur les données dapprentissage et de validation.
# 
# En revanche nous obtenons un très mauvais taux de bonne prédiction sur les données de test (62,50%). 

# In[ ]:


def predict_image(img_path): #fonction qui télécharge une image, la normalise et prédit si elle 
#appartient à la classe chihuahua ou à la classe muffin
    img = load_img(img_path,target_size=(img_height, img_width))
    x = img_to_array(img)
    x = x/255
    x = x.reshape((1,) + x.shape)
    plt.imshow(img)
    print([name for name, age in train_generator.class_indices.items() if age == model.predict_classes(x)[0]][0])


# In[109]:


#test sur une image de notra training set
predict_image('gdrive/My Drive/Data/chihuahua-vs-muffin/train/chihuahua/ActiOn_1.jpg') 


# In[110]:


predict_image('gdrive/My Drive/Data/chihuahua-vs-muffin/test/chihuahua/image1.jpg')


# In[111]:


predict_image('gdrive/My Drive/Data/chihuahua-vs-muffin/test/chihuahua/image2.jpg')


# In[112]:


predict_image('gdrive/My Drive/Data/chihuahua-vs-muffin/test/chihuahua/image3.jpg')


# In[113]:


predict_image('gdrive/My Drive/Data/chihuahua-vs-muffin/test/chihuahua/image4.jpg')


# In[114]:


predict_image('gdrive/My Drive/Data/chihuahua-vs-muffin/test/chihuahua/image5.jpg')


# In[115]:


predict_image('gdrive/My Drive/Data/chihuahua-vs-muffin/test/chihuahua/image6.jpg')


# In[116]:


predict_image('gdrive/My Drive/Data/chihuahua-vs-muffin/test/chihuahua/image7.jpg')


# In[117]:


predict_image('gdrive/My Drive/Data/chihuahua-vs-muffin/test/chihuahua/image8.jpg')


# In[118]:


predict_image('gdrive/My Drive/Data/chihuahua-vs-muffin/test/muffin/image9.jpg')


# In[119]:


predict_image('gdrive/My Drive/Data/chihuahua-vs-muffin/test/muffin/image10.jpg')


# In[120]:


predict_image('gdrive/My Drive/Data/chihuahua-vs-muffin/test/muffin/image11.jpg')


# In[121]:


predict_image('gdrive/My Drive/Data/chihuahua-vs-muffin/test/muffin/image12.jpg')


# In[122]:


predict_image('gdrive/My Drive/Data/chihuahua-vs-muffin/test/muffin/image13.jpg')


# In[123]:


predict_image('gdrive/My Drive/Data/chihuahua-vs-muffin/test/muffin/image14.jpg')


# In[124]:


predict_image('gdrive/My Drive/Data/chihuahua-vs-muffin/test/muffin/image15.jpg')


# In[125]:


predict_image('gdrive/My Drive/Data/chihuahua-vs-muffin/test/muffin/image16.jpg')


# L'utilisation de la fonction predict_image() sur les jeux de données de test confirme le résultat obtenus ci-dessus. En effet, la fonction donne des résultats pas du tout satisfaisants sur nos données de test (prédictions justes : 10/16, soit 62,5%, ce qui correspond exactement à l'accuracy score). Testons maintenant la 2ème méthode qui consiste à utiliser un jeu de données d'apprentissage déjà entrainé (transfert learning). 

# # 2ème méthode : Méthode Bottleneck Information
# 
# 

# Selon l'inventeur de cette méthode, Naftali Tishby (méthode également appelée principe du goulot d'étrangelement de l'information), les réseaux de neurones profonds sont capables d'apprendre grâce à l'information bottleneck. <br>
# 
# Un réseau traite les multiples données d’entrée renfermant de très nombreux détails en extrayant l’information comme s’il les faisait passer par un goulot de bouteille. Ainsi, il ne retient que les informations les plus importantes en fonction des concepts généraux. <br>
# 
# Cette méthode est conçue pour trouver le meilleur compromis entre précision et complexité ( compression ) lors de la synthèse (par exemple, en clustering ) d'une variable aléatoire X , étant donné une distribution de probabilité conjointe p (X, Y) entre X et une variable pertinente observée Y. <br>
# 
# Dans ce TP nous allons utiliser vgg16 (une version du réseau de neurones convolutif très connu appelé VGG-Net) fournie par Keras, qui a déjà été entrainé sur les 14 millions d'images présentes dans la base de données Imagenet, et qui permet de faire du transfert learning. <br>
# <br>
# <strong> Aparté : Qu'est ce que vgg16 et comment est constitué son architecture </strong> <br>
# VGG-16 est constitué de plusieurs couches, dont 13 couches de convolution et 3 couches fully-connected. Il doit donc apprendre les poids de 16 couches. <br>
# 
# Il prend en entrée une image en couleurs de taille 224  ×
#  224 px et la classifie dans une des 1000 classes. Il renvoie donc un vecteur de taille 1000, qui contient les probabilités d'appartenance à chacune des classes. <br>
# 
# Chaque couche de convolution utilise des filtres en couleurs de taille 3  ×
# 3 px, déplacés avec un pas de 1 pixel. Le zero-padding vaut 1 pixel afin que les volumes en entrée aient les mêmes dimensions en sortie. Le nombre de filtres varie selon le "bloc" dans lequel la couche se trouve. De plus, un paramètre de biais est introduit dans le produit de convolution pour chaque filtre.
# 
# Chaque couche de convolution a pour fonction d'activation une ReLU. Autrement dit, il y a toujours une couche de correction ReLU après une couche de convolution.
# 
# L'opération de pooling est réalisée avec des cellules de taille 2 ×
#  2 px et un pas de 2 px – les cellules ne se chevauchent donc pas.
# 
# Les deux premières couches fully-connected calculent chacune un vecteur de taille 4096, et sont chacune suivies d'une couche ReLU. La dernière renvoie le vecteur de probabilités de taille 1000 (le nombre de classes) en appliquant la fonction softmax. De plus, ces trois couches utilisent un paramètre de biais pour chaque élément du vecteur en sortie. <br>
# 
# Plus de détails : https://openclassrooms.com/fr/courses/4470531-classez-et-segmentez-des-donnees-visuelles/5097666-tp-implementez-votre-premier-reseau-de-neurones-avec-keras
# 
# <br> Pour récapituler, la méthode Bottleneck Information est une méthode de transfert learning qui consiste à utiliser le réseau de neurones comme une variable fixe d’extraction de caractéristiques et à l’appliquer telle quelle à notre dataset. Voici les étapes :
# 
# - On enlève la dernière couche, i.e. fully connected, du réseau,
# - On « gèle » les poids du modèle et nous les utilisons comme une variable fixe d’extrait,
# - On extrait alors les 4096 dimensionnels codes grâce à la variable fixée pour toutes les images,
# - On entraîne une classification linéaire pour le nouveau dataset (e.g. SVM linéaire ou UMAP comme dans nore cas ci-dessous).

# In[ ]:


from tensorflow.keras import applications
#Les applications Keras sont des modèles d'apprentissage en profondeur qui sont mis à disposition avec 
#des poids pré-formés. Ces modèles peuvent être utilisés pour la prédiction, l'extraction de fonctionnalités 
#(exemple bottleneck information) et le réglage fin (fine tuning)
#Plus de détails : https://translate.google.com/translate?hl=fr&sl=en&u=http://keras.io/applications/&prev=search
from tensorflow.keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt


# In[127]:


import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, GlobalMaxPooling2D
from keras import applications
from keras.applications.vgg16 import preprocess_input
# preprocess_imput : permet d'appliquer les mêmes pré-traitements que ceux utilisés sur l'ensemble d'apprentissage 
#lors du pré-entraînement. 

#from utils.canvas_embedding import *
from tensorflow.keras.utils import *

from IPython.core.display import HTML
HTML("<style>.container { width:95% !important; }</style>")


# In[148]:


get_ipython().run_cell_magic('time', '', "img_width, img_height = 299,299\n#Précision du répertoire où se trouvent nos dataset d'apprentissage, de test et de validation\ntrain_data_dir = 'gdrive/My Drive/Data/chihuahua-vs-muffin/train'\nvalidation_data_dir = 'gdrive/My Drive/Data/chihuahua-vs-muffin/validation'\ntest_data_dir = 'gdrive/My Drive/Data/chihuahua-vs-muffin/test'\n\ndatagen = ImageDataGenerator()\ngenerator_train = datagen.flow_from_directory(shuffle=False, batch_size=32,target_size=(img_height,img_width), directory=train_data_dir, class_mode='categorical')\ngenerator_validation = datagen.flow_from_directory(shuffle=False, batch_size=32,target_size=(img_height,img_width), directory=validation_data_dir, class_mode='categorical')\ngenerator_test = datagen.flow_from_directory(shuffle=False, batch_size=32,target_size=(img_height,img_width), directory=test_data_dir, class_mode='categorical')")


# In[ ]:


#création du mod§le VGG16
modelVGG16 = applications.VGG16(include_top=False, weights='imagenet', pooling='max')


# In[150]:


#pre processing des training set
%%time
train_data = []
train_data_processed = []
train_target = []

for i in range(len(generator_train)):
    print(i)
    (data,target) = generator_train.next()
    train_data.append(data)
    train_data_processed.append(modelVGG16.predict(preprocess_input(data)))
    train_target.append(target)


# In[151]:


#pre processing des validation set

%%time
validation_data = []
validation_data_processed = []
validation_target = []

for i in range(len(generator_validation)):
    print(i)
    (data,target) = generator_validation.next()
    validation_data.append(data)
    validation_data_processed.append(modelVGG16.predict(preprocess_input(data)))
    validation_target.append(target)


# In[152]:


#pre processing des test set

%%time
test_data = []
test_data_processed = []
test_target = []

for i in range(len(generator_test)):
    (data,target) = generator_test.next()
    test_data.append(data)
    test_data_processed.append(modelVGG16.predict(preprocess_input(data)))
    test_target.append(target)


# In[153]:



import numpy as np

train_data = np.concatenate(train_data)
validation_data = np.concatenate(validation_data)
test_data = np.concatenate(test_data)


train_data_processed = np.concatenate(train_data_processed)
validation_data_processed = np.concatenate(validation_data_processed)
test_data_processed = np.concatenate(test_data_processed)


train_target = np.concatenate(train_target)
validation_target = np.concatenate(validation_target)
test_target = np.concatenate(test_target)


print(train_target.shape)
print(validation_target.shape)
print(test_target.shape)


# # SVC

# In[155]:


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

train_target_sklearn = np.argmax(train_target,axis=1)
validation_target_sklearn = np.argmax(validation_target,axis=1)
test_target_sklearn = np.argmax(test_target,axis=1)

clf = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC())])

clf.fit(train_data_processed,train_target_sklearn)


# In[156]:


clf.score(train_data_processed,train_target_sklearn)


# In[158]:


clf.score(validation_data_processed,validation_target_sklearn)


# In[157]:


clf.score(test_data_processed,test_target_sklearn)


# En appliquant un algorithme de SVC sur nos données, après avoir utilisé un modèle VGG16, nous obtenons un taux de bonne prédiction de : 
# - 100% sur les données d'apprentissage (ce qui est plus satisfaisant que les 99,29% obtenus en apprenant nous-même notre modèle) 
# - 98,58% sur les données de validation (ce qui est plus satisfaisant que les 91,49% obtenus en apprenant nous-même notre modèle)
# - 87,50% sur les données de test (ce qui est nettement plus satisfaisant que les 62,50% obtenus en apprenant nous-même notre modèle)

# # UMAP

# UMAP (Uniform Manifold Approximation and Projection) est une méthode de réduction des dimensions qui concurrence le t-SNE. L'UMAP est construit à partir d'un cadre théorique basé sur la géométrie riemannienne et la topologie algébrique. Le résultat est un algorithme évolutif pratique qui s'applique aux données du monde réel. L'algorithme UMAP est compétitif avec t-SNE pour la qualité de la visualisation et préserve sans doute une plus grande partie de la structure globale avec des performances d'exécution supérieures. En outre, UMAP n'a aucune restriction de calcul sur l'intégration de la dimension, ce qui la rend viable en tant que technique de réduction de dimension à usage général pour l'apprentissage automatique.

# In[ ]:


data = np.concatenate((train_data,validation_data))
data_processed = np.concatenate((train_data_processed,validation_data_processed))
targets_sklearn = np.concatenate((train_target_sklearn,validation_target_sklearn))


# In[162]:


import umap
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.sparse import csgraph

embedding = umap.UMAP()
data_umaped = embedding.fit_transform(data_processed)
plt.figure(figsize=(15,15))
plt.scatter(data_umaped[:,1],data_umaped[:,0],c=targets_sklearn)
plt.gca().invert_yaxis()


# Nous pouvons constater avec le graphique ci-dessus que nos 2 classes sont effectivement bien distinctes et bien séparées, à l'exception de quelques petits points violets qui apparaissent dans la classe jaune. Ce résultat est cohérent avec les taux de bonne classification que nous avons obtenu ci-dessus avec le SVC.

# In[ ]:


from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image as pil_image
import numpy as np
import matplotlib.pyplot as plt

def min_resize(img, size):
    w, h = map(float, img.shape[:2])
    if min([w, h]) != size:
        if w <= h:
            img = array_to_img(img).resize((int(round((h/w)*size)), int(size)),pil_image.NEAREST)
        else:
            img = array_to_img(img).resize((int(size), int(round((w/h)*size))),pil_image.NEAREST)
    return img_to_array(img)/255

def image_scatter(features_tsned, images, img_res, res=4000, cval=1.):
    images = [min_resize(image, img_res) for image in images]
    max_width = max([image.shape[0] for image in images])
    max_height = max([image.shape[1] for image in images])

    f2d = features_tsned

    xx = f2d[:, 0]
    yy = f2d[:, 1]
    x_min, x_max = xx.min(), xx.max()
    y_min, y_max = yy.min(), yy.max()
    # Fix the ratios
    sx = (x_max-x_min)
    sy = (y_max-y_min)
    if sx > sy:
        res_x = int(sx/float(sy)*res)
        res_y = res
    else:
        res_x = res
        res_y = int(sy/float(sx)*res)

    canvas = np.ones((res_x+max_width, res_y+max_height, 3))*cval
    x_coords = np.linspace(x_min, x_max, res_x)
    y_coords = np.linspace(y_min, y_max, res_y)
    for x, y, image in zip(xx, yy, images):
        w, h = image.shape[:2]
        x_idx = np.argmin((x_coords - x)**2)
        y_idx = np.argmin((y_coords - y)**2)
        canvas[x_idx:x_idx+w, y_idx:y_idx+h] = image
    return canvas

def save_embedding(canvas,filename,size):
    fig = plt.figure(figsize=(size,size))
    plt.imshow(canvas)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(filename, bbox_inches = 'tight',pad_inches = 0)
    plt.close(fig)

def embedding(vis_data,ims,S,s):
    vis_data = vis_data - np.min(vis_data,0)
    data_norm = vis_data / np.max(vis_data,0)

    canvas = np.zeros((S,S, 3))
    i = 0
    for image in ims:
        a = np.ceil(data_norm[i] * (S-s)+1)[0]
        b = np.ceil(data_norm[i] * (S-s)+1)[1]
        aa = int(a-(((a-1) % s)+1))
        bb = int(b-(((b-1) % s)+1))
        if canvas[aa,bb,1] == 0:
            canvas[aa:aa+s, bb:bb+s] = min_resize(image, s)
        i+=1
    return canvas

def square_embedding(vis_data,ims,S,s):
    vis_data = vis_data - np.min(vis_data,0)
    data_norm = vis_data / np.max(vis_data,0)

    N = len(ims)
    canvas_square = np.zeros((S,S, 3))
    used = np.zeros(N,dtype=int)

    qq = len(np.arange(0,S,s))
    abes = np.zeros((qq*qq,2))
    i=0
    for a in np.arange(0,S,s):
        for b in np.arange(0,S,s):
            abes[i,:] = [a,b]
            i=i+1

    for i in range(abes.shape[0]):
        a = int(abes[i,0])
        b = int(abes[i,1])
        xf = (a-1)/S
        yf = (b-1)/S
        dd = np.sum(np.power(data_norm - np.array((xf,yf)),2),axis=1)
        dd[np.where(used == True)[0].tolist()] = float('Inf')
        di = np.argmin(dd)
        used[di] = True
        canvas_square[a:a+s, b:b+s] = min_resize(ims[di], s)
    return canvas_square


# In[164]:


canvas_umap = image_scatter(data_umaped,data,100)
plt.figure(figsize=(40,40))
plt.imshow(canvas_umap)


# In[ ]:


import matplotlib.pyplot as plt
save_embedding(canvas_umap,'chihuahua_muffin.png',40)


# VGG16 est un modèle de réseau de neurones convolutionnel proposé par K. Simonyan et A. Zisserman de l'Université d'Oxford dans l'article «Réseaux de convolution très profonds pour la reconnaissance d'images à grande échelle». Le modèle atteint une précision de test de 92,7% dans le top 5 dans ImageNet. VGG16 était l'un des célèbres modèles soumis à l' ILSVRC-2014. Ce modèle apporte une amélioration par rapport à AlexNet en remplaçant les grands filtres de taille de noyau (11 et 5 dans la première et la deuxième couche convolutionnelle respectivement) par plusieurs filtres de taille de noyau 3 × 3 l'un après l'autre. 
# <br>
# <br>
# Cependant il existe d'autres modèles de réseaux de neurones qui ont succédé à VGG 16 (et qui sont disponibles dans Applications de Keras). A titre d'exemple, le modèle ResNet qui a remporté le défi d'ImageNet en 2015. La percée fondamentale avec ResNet a été de permettre de former avec succès des réseaux neuronaux extrêmement profonds avec plus de 150 couches. Avant la formation de ResNet, les réseaux neuronaux très profonds étaient difficiles à former en raison du problème de la disparition des gradients. 
# <br>
# <br>
# Dans la suite de ce tp, pour cette 2ème méthode, nous allons tester différents types de modèles de réseaux de neurones tels que ResNet50, ResNet152, DenseNet, et nous comparerons les scores obtenus avec ceux obtenus en utilisant VGG16.

# # Test avec ResNet 50

# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
from tensorflow.keras import applications
from tensorflow.keras.applications.resnet import preprocess_input

import matplotlib.pyplot as plt


# In[237]:


import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, GlobalMaxPooling2D
from keras import applications
from keras.applications.resnet import preprocess_input
#from utils.canvas_embedding import *

from IPython.core.display import HTML
HTML("<style>.container { width:95% !important; }</style>")


# In[238]:


get_ipython().run_cell_magic('time', '', "img_width, img_height = 299,299\n#Précision du répertoire où se trouvent nos dataset d'apprentissage, de test et de validation\ntrain_data_dir = 'gdrive/My Drive/Data/chihuahua-vs-muffin/train'\nvalidation_data_dir = 'gdrive/My Drive/Data/chihuahua-vs-muffin/validation'\ntest_data_dir = 'gdrive/My Drive/Data/chihuahua-vs-muffin/test'\n\ndatagen = ImageDataGenerator()\ngenerator_train = datagen.flow_from_directory(shuffle=False, batch_size=32,target_size=(img_height,img_width), directory=train_data_dir, class_mode='categorical')\ngenerator_validation = datagen.flow_from_directory(shuffle=False, batch_size=32,target_size=(img_height,img_width), directory=validation_data_dir, class_mode='categorical')\ngenerator_test = datagen.flow_from_directory(shuffle=False, batch_size=32,target_size=(img_height,img_width), directory=test_data_dir, class_mode='categorical')")


# In[ ]:


#Création du modèle ResNet50
modelResNet50 = applications.resnet.ResNet50(include_top=False, weights='imagenet', pooling='max')


# In[248]:


#Pre processing des données d'apprentissage, de validation et de test

%%time
train_data = []
train_data_processed = []
train_target = []

for i in range(len(generator_train)):
    print(i)
    (data,target) = generator_train.next()
    train_data.append(data)
    train_data_processed.append(modelResNet152.predict(preprocess_input(data)))
    train_target.append(target)

validation_data = []
validation_data_processed = []
validation_target = []

for i in range(len(generator_validation)):
    print(i)
    (data,target) = generator_validation.next()
    validation_data.append(data)
    validation_data_processed.append(modelResNet152.predict(preprocess_input(data)))
    validation_target.append(target)

test_data = []
test_data_processed = []
test_target = []

for i in range(len(generator_test)):
    (data,target) = generator_test.next()
    test_data.append(data)
    test_data_processed.append(modelResNet152.predict(preprocess_input(data)))
    test_target.append(target)


# In[249]:


#Concaténation des données

import numpy as np

train_data = np.concatenate(train_data)
validation_data = np.concatenate(validation_data)
test_data = np.concatenate(test_data)


train_data_processed = np.concatenate(train_data_processed)
validation_data_processed = np.concatenate(validation_data_processed)
test_data_processed = np.concatenate(test_data_processed)


train_target = np.concatenate(train_target)
validation_target = np.concatenate(validation_target)
test_target = np.concatenate(test_target)


print(train_target.shape)
print(validation_target.shape)
print(test_target.shape)


# # SVC

# In[250]:


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

train_target_sklearn_50 = np.argmax(train_target_50,axis=1)
validation_target_sklearn_50 = np.argmax(validation_target_50,axis=1)
test_target_sklearn_50 = np.argmax(test_target_50,axis=1)

clf = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC())])

clf.fit(train_data_processed,train_target_sklearn)


# In[252]:


clf.score(train_data_processed,train_target_sklearn)


# In[253]:


clf.score(validation_data_processed,validation_target_sklearn)


# In[254]:


clf.score(test_data_processed,test_target_sklearn)


# En appliquant un algorithme de SVC sur nos données, après avoir utilisé un modèle ResNet50, nous obtenons un taux de bonne prédiction de : 
# - 100% sur les données d'apprentissage (même score qu'avec VGG16 et plus satisfaisant que les 99,29% obtenus en entrainant nous même notre propre modèle) 
# - 99,29% sur les données de validation (ce qui est plus satisfaisant que les 98,58% obtenus avec VGG16 et les 91,49% obtenus en entrainant nous même notre propre modèle)
# - 100% sur les données de test (ce qui est nettement plus satisfaisant que les 87,50% obtenus avec VGG16 et les 62,50% obtenus en entrainant nous même notre propre modèle)

# # UMAP

# In[ ]:


data = np.concatenate((train_data,validation_data))
data_processed = np.concatenate((train_data_processed,validation_data_processed))
targets_sklearn = np.concatenate((train_target_sklearn,validation_target_sklearn))


# In[258]:


import umap
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.sparse import csgraph

embedding = umap.UMAP()
data_umaped = embedding.fit_transform(data_processed)
plt.figure(figsize=(15,15))
plt.scatter(data_umaped[:,1],data_umaped[:,0],c=targets_sklearn)
plt.gca().invert_yaxis()


# Nous pouvons constater avec ce graphique obtenu en utilisant la fonction UMAP que les 2 classes sont mieux séparés qu'avec VGG16, et qu'il n'y a que deux points violets dans la classe jaune, ce qui est cohérent avec le score obtenu avec SVC sur le dataset de validation (99%). 
# <br>
# Maintenant, testons le modèles ResNet152 qui comme son nom l'indique est un modèle de réseaux de neurones extrêmement profond constitué de 152 couches et comparons le score obtenu (qui devrait en principe être meilleur) avec ceux obtenus en utilisant VGG16 et ResNet50. 
# <br>

# # Test avec ResNet152

# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
from tensorflow.keras import applications
from tensorflow.keras.applications.resnet import preprocess_input

import matplotlib.pyplot as plt


# In[260]:


import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, GlobalMaxPooling2D
from keras import applications
from keras.applications.resnet import preprocess_input
#from utils.canvas_embedding import *

from IPython.core.display import HTML
HTML("<style>.container { width:95% !important; }</style>")


# In[261]:


get_ipython().run_cell_magic('time', '', "img_width, img_height = 299,299\n#Précision du répertoire où se trouvent nos dataset d'apprentissage, de test et de validation\ntrain_data_dir = 'gdrive/My Drive/Data/chihuahua-vs-muffin/train'\nvalidation_data_dir = 'gdrive/My Drive/Data/chihuahua-vs-muffin/validation'\ntest_data_dir = 'gdrive/My Drive/Data/chihuahua-vs-muffin/test'\n\ndatagen = ImageDataGenerator()\ngenerator_train = datagen.flow_from_directory(shuffle=False, batch_size=32,target_size=(img_height,img_width), directory=train_data_dir, class_mode='categorical')\ngenerator_validation = datagen.flow_from_directory(shuffle=False, batch_size=32,target_size=(img_height,img_width), directory=validation_data_dir, class_mode='categorical')\ngenerator_test = datagen.flow_from_directory(shuffle=False, batch_size=32,target_size=(img_height,img_width), directory=test_data_dir, class_mode='categorical')")


# In[ ]:


#Création du modèle ResNet152
modelResNet152 = applications.resnet.ResNet152(include_top=False, weights='imagenet', pooling='max')


# In[264]:


#Pre processing des données d'apprentissage, de validation et de test

%%time
train_data = []
train_data_processed = []
train_target = []

for i in range(len(generator_train)):
    print(i)
    (data,target) = generator_train.next()
    train_data.append(data)
    train_data_processed.append(modelResNet152.predict(preprocess_input(data)))
    train_target.append(target)

validation_data = []
validation_data_processed = []
validation_target = []

for i in range(len(generator_validation)):
    print(i)
    (data,target) = generator_validation.next()
    validation_data.append(data)
    validation_data_processed.append(modelResNet152.predict(preprocess_input(data)))
    validation_target.append(target)

test_data = []
test_data_processed = []
test_target = []

for i in range(len(generator_test)):
    (data,target) = generator_test.next()
    test_data.append(data)
    test_data_processed.append(modelResNet152.predict(preprocess_input(data)))
    test_target.append(target)


# In[265]:


#Concaténation des données

import numpy as np

train_data = np.concatenate(train_data)
validation_data = np.concatenate(validation_data)
test_data = np.concatenate(test_data)


train_data_processed = np.concatenate(train_data_processed)
validation_data_processed = np.concatenate(validation_data_processed)
test_data_processed = np.concatenate(test_data_processed)


train_target = np.concatenate(train_target)
validation_target = np.concatenate(validation_target)
test_target = np.concatenate(test_target)


print(train_target.shape)
print(validation_target.shape)
print(test_target.shape)


# # SVC

# In[266]:


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

train_target_sklearn = np.argmax(train_target,axis=1)
validation_target_sklearn = np.argmax(validation_target,axis=1)
test_target_sklearn = np.argmax(test_target,axis=1)

clf = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC())])

clf.fit(train_data_processed,train_target_sklearn)


# In[267]:


clf.score(train_data_processed,train_target_sklearn)


# In[268]:


clf.score(validation_data_processed,validation_target_sklearn)


# In[269]:


clf.score(test_data_processed,test_target_sklearn)


# Nous obtenons exactement le même score que celui obtenu avec ResNet50 sur les données d'apprentissage, de test et de validation.
# <br>
# Cependant, des travaux publiés en 2016 ont montré que les réseaux convolutionnels peuvent être sensiblement plus profonds, plus précis et plus efficaces s'ils contiennent des connexions plus courtes entre les couches proches de l'entrée et celles proches de la sortie. C'est ainsi qu'on été créé les modèles de réseau de neurones DenseNet. DenseNet connecte chaque couche à toutes les autres couches de manière directe. Alors que les réseaux convolutionnels traditionnels avec L couches ont L connexions - une entre chaque couche et sa couche suivante - le réseau DenseNet a L (L + 1)/2 connexions directes. Pour chaque couche, les cartes d'entités de toutes les couches précédentes sont utilisées comme entrées et ses propres cartes d'entités sont utilisées comme entrées dans toutes les couches suivantes. 
# <br>
# <br>
# Les DenseNets présentent plusieurs avantages convaincants:
# * ils atténuent le problème de la disparition du gradient
# * renforcent la propagation des entités
# * encouragent la réutilisation des entités et réduisent considérablement le nombre de paramètres.
# <br>
# <br>
# Les DenseNets obtiennent des améliorations significatives par rapport à l'état de l'art sur des tâches de référence de reconnaissance d'image telles que ImageNet, CIFAR-10), tout en nécessitant moins de calculs pour atteindre des performances élevées.
# <br>
# Dans la suite de ce TP, nous allons tester la méthode Bottleneck Information avec un modèle DenseNet qui devrait en principe donner un meilleur score que VGG16 et ResNet.

# # Test avec DenseNet

# In[ ]:


# Test avec DenseNet

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
from tensorflow.keras import applications
from tensorflow.keras.applications.densenet import preprocess_input

import matplotlib.pyplot as plt


# In[271]:


import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, GlobalMaxPooling2D
from keras import applications
from keras.applications.densenet import preprocess_input
#from utils.canvas_embedding import *

from IPython.core.display import HTML
HTML("<style>.container { width:95% !important; }</style>")


# In[272]:


get_ipython().run_cell_magic('time', '', "img_width, img_height = 299,299\n#Précision du répertoire où se trouvent nos dataset d'apprentissage, de test et de validation\ntrain_data_dir = 'gdrive/My Drive/Data/chihuahua-vs-muffin/train'\nvalidation_data_dir = 'gdrive/My Drive/Data/chihuahua-vs-muffin/validation'\ntest_data_dir = 'gdrive/My Drive/Data/chihuahua-vs-muffin/test'\n\ndatagen = ImageDataGenerator()\ngenerator_train = datagen.flow_from_directory(shuffle=False, batch_size=32,target_size=(img_height,img_width), directory=train_data_dir, class_mode='categorical')\ngenerator_validation = datagen.flow_from_directory(shuffle=False, batch_size=32,target_size=(img_height,img_width), directory=validation_data_dir, class_mode='categorical')\ngenerator_test = datagen.flow_from_directory(shuffle=False, batch_size=32,target_size=(img_height,img_width), directory=test_data_dir, class_mode='categorical')")


# In[273]:


# Creation du modele DenseNet
modelDenseNet201 = applications.densenet.DenseNet201(include_top=False, weights='imagenet', pooling='max')


# In[274]:


#Pre processing des données d'apprentissage, de validation et de test

%%time
train_data = []
train_data_processed = []
train_target = []

for i in range(len(generator_train)):
    print(i)
    (data,target) = generator_train.next()
    train_data.append(data)
    train_data_processed.append(modelDenseNet201.predict(preprocess_input(data)))
    train_target.append(target)

validation_data = []
validation_data_processed = []
validation_target = []

for i in range(len(generator_validation)):
    print(i)
    (data,target) = generator_validation.next()
    validation_data.append(data)
    validation_data_processed.append(modelDenseNet201.predict(preprocess_input(data)))
    validation_target.append(target)

test_data = []
test_data_processed = []
test_target = []

for i in range(len(generator_test)):
    (data,target) = generator_test.next()
    test_data.append(data)
    test_data_processed.append(modelDenseNet201.predict(preprocess_input(data)))
    test_target.append(target)


# In[275]:


#Concaténation des données

import numpy as np

train_data = np.concatenate(train_data)
validation_data = np.concatenate(validation_data)
test_data = np.concatenate(test_data)


train_data_processed = np.concatenate(train_data_processed)
validation_data_processed = np.concatenate(validation_data_processed)
test_data_processed = np.concatenate(test_data_processed)


train_target = np.concatenate(train_target)
validation_target = np.concatenate(validation_target)
test_target = np.concatenate(test_target)


print(train_target.shape)
print(validation_target.shape)
print(test_target.shape)


# # SVC

# In[276]:


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

train_target_sklearn = np.argmax(train_target,axis=1)
validation_target_sklearn = np.argmax(validation_target,axis=1)
test_target_sklearn = np.argmax(test_target,axis=1)

clf = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC())])

clf.fit(train_data_processed,train_target_sklearn)


# In[277]:


clf.score(train_data_processed,train_target_sklearn)


# In[278]:


clf.score(validation_data_processed,validation_target_sklearn)


# In[279]:


clf.score(test_data_processed,test_target_sklearn)


# Comme attendu, en appliquant un algorithme de SVC sur nos données, après avoir utilisé un modèle DenseNet, nous obtenons des meilleurs scores (100% sur les données d'apprentissage, 100% sur les données de validation et 100% sur les données de test) ce qui est nettement mieux que ceux obtenus avec VGG16, ResNet50 et ResNet152. 
# 
# Tesons maintenant le modèle Inception Resnet V2 qui a le 2ème score le plus élevé dans le Top-1 Accuracy sur Keras (https://keras.io/applications/),
# 
# Testons maintenant la 3ème méthode (fine-tuning).
# 

# # Test avec InceptionResNetV2

# In[ ]:


# Test avec InceptionResNetV2

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
from tensorflow.keras import applications
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input

import matplotlib.pyplot as plt


# In[286]:


import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, GlobalMaxPooling2D
from keras import applications
from keras.applications.inception_resnet_v2 import preprocess_input
#from utils.canvas_embedding import *

from IPython.core.display import HTML
HTML("<style>.container { width:95% !important; }</style>")


# In[287]:


get_ipython().run_cell_magic('time', '', "img_width, img_height = 299,299\n#Précision du répertoire où se trouvent nos dataset d'apprentissage, de test et de validation\ntrain_data_dir = 'gdrive/My Drive/Data/chihuahua-vs-muffin/train'\nvalidation_data_dir = 'gdrive/My Drive/Data/chihuahua-vs-muffin/validation'\ntest_data_dir = 'gdrive/My Drive/Data/chihuahua-vs-muffin/test'\n\ndatagen = ImageDataGenerator()\ngenerator_train = datagen.flow_from_directory(shuffle=False, batch_size=32,target_size=(img_height,img_width), directory=train_data_dir, class_mode='categorical')\ngenerator_validation = datagen.flow_from_directory(shuffle=False, batch_size=32,target_size=(img_height,img_width), directory=validation_data_dir, class_mode='categorical')\ngenerator_test = datagen.flow_from_directory(shuffle=False, batch_size=32,target_size=(img_height,img_width), directory=test_data_dir, class_mode='categorical')")


# In[289]:


# Creation du modele Inception Resnet V2
modelInception_resnet_v2 = applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet', pooling='max')


# In[290]:


#Pre processing des données d'apprentissage, de validation et de test

%%time
train_data = []
train_data_processed = []
train_target = []

for i in range(len(generator_train)):
    print(i)
    (data,target) = generator_train.next()
    train_data.append(data)
    train_data_processed.append(modelInception_resnet_v2.predict(preprocess_input(data)))
    train_target.append(target)

validation_data = []
validation_data_processed = []
validation_target = []

for i in range(len(generator_validation)):
    print(i)
    (data,target) = generator_validation.next()
    validation_data.append(data)
    validation_data_processed.append(modelInception_resnet_v2.predict(preprocess_input(data)))
    validation_target.append(target)

test_data = []
test_data_processed = []
test_target = []

for i in range(len(generator_test)):
    (data,target) = generator_test.next()
    test_data.append(data)
    test_data_processed.append(modelInception_resnet_v2.predict(preprocess_input(data)))
    test_target.append(target)


# In[291]:


#Concaténation des données

import numpy as np

train_data = np.concatenate(train_data)
validation_data = np.concatenate(validation_data)
test_data = np.concatenate(test_data)


train_data_processed = np.concatenate(train_data_processed)
validation_data_processed = np.concatenate(validation_data_processed)
test_data_processed = np.concatenate(test_data_processed)


train_target = np.concatenate(train_target)
validation_target = np.concatenate(validation_target)
test_target = np.concatenate(test_target)


print(train_target.shape)
print(validation_target.shape)
print(test_target.shape)


# # SVC

# In[292]:


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

train_target_sklearn = np.argmax(train_target,axis=1)
validation_target_sklearn = np.argmax(validation_target,axis=1)
test_target_sklearn = np.argmax(test_target,axis=1)

clf = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC())])

clf.fit(train_data_processed,train_target_sklearn)


# In[293]:


clf.score(train_data_processed,train_target_sklearn)


# In[294]:


clf.score(validation_data_processed,validation_target_sklearn)


# In[295]:


clf.score(test_data_processed,test_target_sklearn)


# Bien que Inception ResNet V2 présente des très bons score dans le top-1 Accuracy et le top-5 Accuracy des applications de Keras (https://keras.io/applications/#nasnet), c'est DenseNet qui donne les meilleurs taux de bonne prédiction dans notre cas.
# <br />
# <br />
# Testons maintenant la 3ème méthode (fine-tuning).

# # 3ème méthode : fine-tuning

# Le réglage fin (ou fine tuning) est un processus qui consiste à prendre un modèle de réseau qui a déjà été formé pour une tâche donnée (exemple ici vgg16) et lui faire effectuer une deuxième tâche similaire.
# 
# En supposant que la tâche d'origine est similaire à la nouvelle tâche, l'utilisation d'un réseau qui a déjà été conçu et formé nous permet de tirer parti de l'extraction de fonctionnalités qui se produit dans les couches avant du réseau sans développer ce réseau d'extraction de fonctionnalités à partir de zéro. 
# <br>
# Les étapes du fine tuning sont :
# <br>
# - La pratique courante consiste à tronquer la dernière couche ( couche softmax) du réseau pré-formé et à la remplacer par notre nouvelle couche softmax correspondant à notre propre problème. Par exemple, un réseau pré-formé sur ImageNet est livré avec une couche softmax avec 1000 catégories.
# <br>
# Si notre tâche est un classement sur 10 catégories, la nouvelle couche softmax du réseau sera de 10 catégories au lieu de 1000 catégories. Nous reportons ensuite la propagation sur le réseau pour affiner les poids pré-entraînés. Il faut s'assurer que la validation croisée soit effectuée afin que le réseau puisse bien se généraliser.
# <br>
# - Utiliser un taux d'apprentissage (learnig rate) plus faible pour former le réseau. Comme nous nous attendons à ce que les poids pré-entraînés soient déjà assez bons par rapport aux poids initialisés au hasard, nous ne voulons pas les déformer trop rapidement et trop. Une pratique courante consiste à choisir un taux d'apprentissage initial 10 fois plus petit que celui utilisé pour l'apprentissage.
# <br>
# - Une fois cela fait, d'autres couches tardives du modèle peuvent être définies comme "trainable = True" afin que dans d'autres epochs SGD (ou autre optimizer) , leurs poids puissent également être ajustés pour la nouvelle tâche. 

# # Test avec VGG16

# In[ ]:


from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import *


# In[ ]:


# Création du modèle de base vgg16 pré-formé
base_model = VGG16(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
#x = Dropout(0.5, name='avg_pool_dropout')(x) 
predictions = Dense(2, activation='softmax')(x) #2 car nous voulons avoir 2 classes en sortie: chihuahua et muffin
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
#rmsprop est un type d'optimiseur qui est recommandé particulièrement pour les réseaux de neurones récurrents. 
#rmsprop se charge de la cross validation


# In[16]:


#Pre processing et augmentation des données en utilisant la fonction ImageGenerator()
%%time
img_width, img_height = 299,299

train_data_dir = 'gdrive/My Drive/Data/chihuahua-vs-muffin/train'
validation_data_dir = 'gdrive/My Drive/Data/chihuahua-vs-muffin/validation'
test_data_dir = 'gdrive/My Drive/Data/chihuahua-vs-muffin/test'

nb_train_samples = 100
nb_validation_samples = 50
nb_test_samples = 20

epochs = 50
batch_size = 32

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

model.fit(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    verbose=2)


# In[17]:


model.evaluate(train_generator,steps=len(train_generator))


# In[18]:


model.evaluate(validation_generator,steps=len(validation_generator))


# In[303]:


model.evaluate(test_generator,steps=len(test_generator))


# Le fine-tuning avec VGG16 donnes des taux de bonne prédiction de :
# - 98,24 % sur les données d'apprentissage
# - 98,58 % sur les données de validation
# - 81,25 % sur les données de test
# 
# Le fine tuning donne de moins bons score bon score par rapport à la méthode bottleneck Information avec VGG 16 (100% sur les données d'apprentissage, 98,58% sur les données de validation et 87,50%).
# 
# Testons maintenant le fine tuning avec ResNet.

# # Test avec ResNet152

# In[ ]:


from tensorflow.keras.applications.resnet import ResNet152
from tensorflow.keras.applications.resnet import preprocess_input
from keras.applications.resnet import preprocess_input


from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import *


# In[20]:


# Création du modèle de base resnet152 pré-formé
base_model = ResNet152(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
#x = Dropout(0.5, name='avg_pool_dropout')(x) 
predictions = Dense(2, activation='softmax')(x) #2 car nous voulons avoir 2 classes en sortie: chihuahua et muffin
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
#rmsprop est un type d'optimiseur qui est recommandé particulièrement pour les réseaux de neurones récurrents. 
#rmsprop se charge de la cross validation


# In[21]:


#Pre processing et augmentation des données en utilisant la fonction ImageGenerator()
%%time
img_width, img_height = 299,299

train_data_dir = 'gdrive/My Drive/Data/chihuahua-vs-muffin/train'
validation_data_dir = 'gdrive/My Drive/Data/chihuahua-vs-muffin/validation'
test_data_dir = 'gdrive/My Drive/Data/chihuahua-vs-muffin/test'

nb_train_samples = 100
nb_validation_samples = 50
nb_test_samples = 20

epochs = 50
batch_size = 32

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

model.fit(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    verbose=2)


# In[22]:


model.evaluate(train_generator,steps=len(train_generator))


# In[23]:


model.evaluate(validation_generator,steps=len(validation_generator))


# In[24]:


model.evaluate(test_generator,steps=len(test_generator))


# Le fine-tuning avec ResNet152 donnes des taux de bonne prédiction de :
# - 95,42 % sur les données d'apprentissage
# - 98,58 % sur les données de validation
# - 50 % sur les données de test
# 
# Le fine tuning donne de moins bons score bon score par rapport à la méthode bottleneck Information avec ResNet152 (100% sur les données d'apprentissage, 99,29% sur les données de validation et 100%)
# 
# Testons maintenant le fine tuning avec DenseNet.

# # Test avec DenseNet

# In[ ]:


from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.applications.densenet import preprocess_input
from keras.applications.densenet import preprocess_input

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import *


# In[26]:


# Création du modèle de base densenet pré-formé
base_model = DenseNet201(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
#x = Dropout(0.5, name='avg_pool_dropout')(x) 
predictions = Dense(2, activation='softmax')(x) #2 car nous voulons avoir 2 classes en sortie: chihuahua et muffin
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
#rmsprop est un type d'optimiseur qui est recommandé particulièrement pour les réseaux de neurones récurrents. 
#rmsprop se charge de la cross validation


# In[27]:


#Pre processing et augmentation des données en utilisant la fonction ImageGenerator()
%%time
img_width, img_height = 299,299

train_data_dir = 'gdrive/My Drive/Data/chihuahua-vs-muffin/train'
validation_data_dir = 'gdrive/My Drive/Data/chihuahua-vs-muffin/validation'
test_data_dir = 'gdrive/My Drive/Data/chihuahua-vs-muffin/test'

nb_train_samples = 100
nb_validation_samples = 50
nb_test_samples = 20

epochs = 50
batch_size = 32

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

model.fit(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    verbose=2)


# In[28]:


model.evaluate(train_generator,steps=len(train_generator))


# In[29]:


model.evaluate(validation_generator,steps=len(validation_generator))


# In[30]:


model.evaluate(test_generator,steps=len(test_generator))


# Le fine-tuning avec DenseNet donnes des très bons taux de bonne prédiction de :
# - 100 % sur les données d'apprentissage
# - 100 % sur les données de validation
# - 100 % sur les données de test
# 
# Ces taux de bonnes prédiction sont identiques à ceux obtenus avec bootleneck Inforlation en utilisant DenseNet.
# 
# Testons maintenant le fine tuning avec Inception Resnet V2.

# # Test avec InceptionResNetV2

# In[ ]:


from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from keras.applications.inception_resnet_v2 import preprocess_input

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import *


# In[32]:


# Création du modèle de base InceptionResNetV2 pré-formé
base_model = InceptionResNetV2(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
#x = Dropout(0.5, name='avg_pool_dropout')(x) 
predictions = Dense(2, activation='softmax')(x) #2 car nous voulons avoir 2 classes en sortie: chihuahua et muffin
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
#rmsprop est un type d'optimiseur qui est recommandé particulièrement pour les réseaux de neurones récurrents. 
#rmsprop se charge de la cross validation


# In[33]:


#Pre processing et augmentation des données en utilisant la fonction ImageGenerator()
%%time
img_width, img_height = 299,299

train_data_dir = 'gdrive/My Drive/Data/chihuahua-vs-muffin/train'
validation_data_dir = 'gdrive/My Drive/Data/chihuahua-vs-muffin/validation'
test_data_dir = 'gdrive/My Drive/Data/chihuahua-vs-muffin/test'

nb_train_samples = 100
nb_validation_samples = 50
nb_test_samples = 20

epochs = 50
batch_size = 32

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

model.fit(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    verbose=2)


# In[34]:


model.evaluate(train_generator,steps=len(train_generator))


# In[35]:


model.evaluate(validation_generator,steps=len(validation_generator))


# In[36]:


model.evaluate(test_generator,steps=len(test_generator))


# Le fine-tuning avec InceptionResNetV2 donnes des taux de bonne prédiction de :
# - 100 % sur les données d'apprentissage
# - 98,58 % sur les données de validation
# - 100 % sur les données de test
# 
# Le fine tuning donne de moins bons score bon score par rapport à la méthode bottleneck Information avec InceptionResNetV2 (100% sur les données d'apprentissage, 99,29% sur les données de validation et 100%)
# 

# # Récapitulatif des scores obtenus avec les 3 méthodes :
# > <strong> 1ère méthode où l'on a entrainé notre propre modèle </strong>
# 
# - 99,29% sur les données d'apprentissage
# - 91,49% sur les données de validation 
# - 62,50% sur les données de test
# 
# > <strong> 2ème méthode : Bottleneck Information </strong>
# 
# * VGG16 :
#  - 100%
#  - 98,58%
#  - 87,5%
# 
# * ResNet50 et ResNet152 :
#  - 100%
#  - 99,29%
#  - 100%
# 
# * DenseNet :
#  - 100%
#  - 100%
#  - 100%
# 
# * Inception ResNet V2 :
#  - 100%
#  - 99,29%
#  - 100%
# 
# > <strong> 3ème méthode : Fine tuning </strong>
# 
# * VGG16 :
#  - 98,24%
#  - 98,58%
#  - 81,25%
# 
# * ResNet50 et ResNet152 :
#  - 95,42%
#  - 98,58%
#  - 50%
# 
# * DenseNet :
#  - 100%
#  - 100%
#  - 100%
# 
# * Inception ResNet V2 :
#  - 100%
#  - 98,58%
#  - 100%
# 

# On peut constater que c'est la 2ème méthode qui donne globalement les meilleurs taux de bonne prédiction parmi les 3 méthodes.
# <br>
# Parmi les 4 modèles de réseaux de neurones déjà pré-entrainé, c'est DenseNet qui donne les meilleurs taux de bonne prédiction comparé à VGG16, ResNet50, ResNet152 et Inception ResNet V2.

# # Conclusion

# - Les méthodes de transfert learning (bottleneck information et fine-tuning) sont nettement préférables à la première méthode où l'on apprend soi-même notre jeu de données, particulièrement quand nous disposons de très peu de données comme dans notre cas.
# 
# - Nous pouvons exploiter le réseau de neurones pré-entraîné de plusieurs façons, en fonction de la taille du jeu de données en entrée et de sa similarité avec celui utilisé lors du pré-entraînement.
# 
# <strong> Stratégie #1 : fine-tuning total </strong>
# <br>
# On remplace la dernière couche fully-connected du réseau pré-entraîné par un classifieur adapté au nouveau problème (SVM, régression logistique...) et initialisé de manière aléatoire. Toutes les couches sont ensuite entraînées sur les nouvelles images. 
# 
# La stratégie #1 doit être utilisée lorsque la nouvelle collection d'images est grande : dans ce cas, on peut se permettre d'entraîner tout le réseau sans courir le risque d'overfitting. De plus, comme les paramètres de toutes les couches (sauf de la dernière) sont initialement ceux du réseau pré-entraîné, la phase d'apprentissage sera faite plus rapidement que si l'initialisation avait été aléatoire.
# 
# <strong>Stratégie #2 : extraction des features</strong>
# <br>
# Cette stratégie consiste à se servir des features du réseau pré-entraîné pour représenter les images du nouveau problème. Pour cela, on retire la dernière couche fully-connected et on fixe tous les autres paramètres. Ce réseau tronqué va ainsi calculer la représentation de chaque image en entrée à partir des features déjà apprises lors du pré-entraînement. On entraîne alors un classifieur, initialisé aléatoirement, sur ces représentations pour résoudre le nouveau problème.
# 
# La stratégie #2 doit être utilisée lorsque la nouvelle collection d'images est petite et similaire aux images de pré-entraînement. En effet, entraîner le réseau sur aussi peu d'images est dangereux puisque le risque d'overfitting est important. De plus, si les nouvelles images ressemblent aux anciennes, elles peuvent alors être représentées par les mêmes features.
# 
# <strong>Stratégie #3 : fine-tuning partiel</strong>
# <br>
# Il s'agit d'un mélange des stratégies #1 et #2 : on remplace à nouveau la dernière couche fully-connected par le nouveau classifieur initialisé aléatoirement, et on fixe les paramètres de certaines couches du réseau pré-entraîné. Ainsi, en plus du classifieur, on entraîne sur les nouvelles images les couches non-fixées, qui correspondent en général aux plus hautes du réseau.
# 
# On utilise cette stratégie lorsque la nouvelle collection d'images est petite mais très différente des images du pré-entraînement. D'une part, comme il y a peu d'images d'entraînement, la stratégie #1 qui consiste à entraîner tout le réseau n'est pas envisageable à cause du risque d'overfitting.
# 
# D'autre part, on élimine également la stratégie #2 puisque les nouvelles images ont très peu de points communs avec les anciennes : utiliser les features du réseau pré-entraîné pour les représenter n'est pas une bonne idée ! Mais souvenez-vous : les features des couches basses sont simples et génériques (donc peuvent se retrouver dans deux images très différentes), tandis que celles des couches hautes sont complexes et spécifiques au problème. Ainsi, la stratégie de fixer les couches basses et d'entraîner le classifieur et les couches hautes constitue un bon compromis.

# 
