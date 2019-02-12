import fitz

#Load PDF
doc = fitz.open('History and Physical.pdf')
pagecount = doc.pageCount
medtext = ['Toprol XL','Digoxin','Lasix','Lisinopril','Spironolactone','Lortab','Excedrin','Protonix','Tylenol',
           'Nitro', 'ASA', 'Aspirin', 'nitroglycerin']
#iterate through each page of document, iterate through term list, add annotation for each term
for pageN in range(pagecount):
    page = doc[pageN]
    for med in medtext:
        text_instances = page.searchFor(med)

        for inst in text_instances:
            highlight = page.addHighlightAnnot(inst)


probtext =['HTN', 'CHF', 'dilated cardiomyopathy', 'polysubstance abuse', 'pain', 'SOB', 'heart failure', 'chest pain', 'chest tightness', 'pressure', 'constant', 'coughing', 'green/brown sputum',
           'breath', 'orthopnea', 'DOE', 'PND', 'abdominal distention', 'peripheral edema',]

for pageN in range(pagecount):
    page = doc[pageN]
    for prob in probtext:
        text_instances = page.searchFor(prob)

        for inst in text_instances:
            highlightprob = page.addHighlightAnnot(inst)
            highlightprob.setColors({"stroke":(0, 1, 1), "fill":(240, 248, 255)})
            highlightprob.update()
           
proctext = ['EKG', 'ECHO', 'Mastectomy', 'stress test', 'chemotherapy', 'XRT', 'BNP', 'C-section']
            
for pageN in range(pagecount):
    page = doc[pageN]
    for proc in proctext:
        text_instances = page.searchFor(proc)

        for inst in text_instances:
            highlightproc = page.addHighlightAnnot(inst)
            highlightproc.setColors({"stroke":(0.5, 0.75, 0), "fill":(240, 248, 255)})
            highlightproc.update()
    zoom_x = 2.0                       # horizontal zoom
    zomm_y = 2.0                       # vertical zoom
    mat = fitz.Matrix(zoom_x, zomm_y)  # zoom factor 2 in each dimension
    pix = page.getPixmap(matrix = mat) # use 'mat' instead of the identity matrix

doc.save("testout.pdf", garbage=4, deflate=True, clean=True)
