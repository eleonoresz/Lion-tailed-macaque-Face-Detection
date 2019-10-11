# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 11:20:39 2019

@author: PateauTech_2
"""

cv2.imwrite('C:/Users/PateauTech_2/Desktop/Positive_Image/image.png',newpart) 

plt.imshow(newpart)
np.min(newpart)
print(newpart)

np.mean(newpart)
npl = list(newpart)
plt.hist(newpart)
newpart.reshape(-1, 1).tolist()
np.size(newpart)/72
l =newpart.tolist()
lst = []
for u in l:
    lst = lst+u
len(lst)
plt.hist(lst,200)
#m.Harris(m.import_images('C:/Users/PateauTech_2/Desktop/Positive_Image/image.png'))

# threshold de significativité déterminé random
for u1 in newpart:
    for u2 in range(len(u1)):
        if u1[u2]<120:
            u1[u2] = 0
plt.imshow(newpart[30:50,15:40])


# dépend de la position dans la photo?

Distance_Array = []
for line in newpart:
    lineValue = []
    Start = False
    for nb in range(10,len(line)-10):
        if line[nb]!=0 and not Start:
            Start = True
            nbStart = nb
        if line[nb]!=0 and Start:
            lineValue.append(nb-nbStart)
            nbStart = nb
    Distance_Array.append(lineValue)
            
# distance du nez
Distance_Array = []
for i in range(30,50):
    lineValue = []
    Start = False
    for j in range(15,40):
        if newpart[i,j]!=0 and not Start:
            Start = True
            nbStart = j
        if newpart[i,j]!=0 and Start:
            lineValue.append(nb-nbStart)
            nbStart = j
    Distance_Array.append(lineValue)


np.max(newpart[30:50,15:40])

# histogram le long de l'axe vertical

for i in range(int(np.size(newpart)/len(newpart))):
    plt.hist(newpart[:,i])
    
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist(newpart,[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()
 #m.imagePath[0]
img = cv2.imread('C:/Users/PateauTech_2/Desktop/Positive_Image/image.png',0)

# create a mask
mask = np.zeros(img.shape[:2], np.uint8)
mask[100:300, 100:400] = 255
masked_img = cv2.bitwise_and(img,img,mask = mask)

# Calculate histogram with mask and without mask
# Check third argument for mask
hist_full = cv2.calcHist([img],[0],None,[256],[0,256])
hist_mask = cv2.calcHist([img],[0],mask,[256],[0,256])

plt.subplot(221), plt.imshow(img, 'gray')
plt.subplot(222), plt.imshow(mask,'gray')
plt.subplot(223), plt.imshow(masked_img, 'gray')
plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
plt.xlim([0,256])

plt.show()
    

img = cv2.imread('C:/Users/PateauTech_2/Desktop/Positive_Image/image.png',0)
plt.hist(img.ravel(),256,[0,256]); plt.show()


#build histograms
TotalHist = []
for nbhist in range(int(ver)):
    TotalHist.append(np.histogram(newpart[:,nbhist]))
    
#put the weight in form
Weight_TotalHist = []
for start in range(0,int(ver)):
    listeinter = []
    for start2 in range(1,len(TotalHist[start][1])):
        listeinter.append((TotalHist[start][1][start2]+TotalHist[start][1][start2-1])/2)
    Weight_TotalHist.append(listeinter)
    