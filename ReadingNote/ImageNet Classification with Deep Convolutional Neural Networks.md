# ImageNet Classification with Deep Convolutional Neural Networks

**{Year}, {Authors}, {Journal Name}**

**{引用格式}**



##  Summary

<!--写完笔记之后最后填，概述文章的内容，以后查阅笔记的时候先看这一段。注：写文章summary切记需要通过自己的思考，用自己的语言描述。忌讳直接Ctrl + c原文-->。



## Research Objective(s)

<!--作者的研究目标是什么-->
* Creat a deep convolutional nerual network to classify the 1.2 million high-resolution images in ImageNet LSVRC-2010 contest into 1000 diffrent classes.
* Have the better performance than previous state-of-the-art




## Background / Problem Statement

<!--研究的背景以及问题陈述：作者需要解决的问题是什么？-->

* Datasets of labeled images were relaticely small
* A model with large learning capacity was needed
* Reduce training time

## Method(s)

<!--作者解决问题的方法/算法是什么？是否基于前人的方法？基于了哪些？-->

* Use CNNs reduce complexity of networks
* Have ReLU Nonlonearity run faster wit fewer iterrations
* Train on Multiple GPUs
* Local response normalization
* Reduce overfitting 
  * data augmentation
	  * generate image translations and horizontal reflections
	  * use PCA altering the intensity of the RGB channels in training images 
  * droupout : reduce complex co-adaptations

## Evaluation

<!--作者如何评估自己的方法？实验的setup是什么样的？感兴趣实验数据和结果有哪些？有没有问题或者可以借鉴的地方？-->

* Train on ILSVRC-2010
  * average the predictions produced from six sparse-coding models trained on different features
  * average the predictions of two classifiers trained on Fisher Vectors computed frome two types of densely-sampled features
* ILSVRC-2012 compitition
* ImageNet



## Conclusion

<!--作者给出了哪些结论？哪些是[strong conclusions](https://www.zhihu.com/search?q=strong+conclusions&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A142802496}), 哪些又是weak的conclusions（即作者并没有通过实验提供evidence，只在discussion中提到；或实验的数据并没有给出充分的evidence）?-->

* [weak conclusions]Using Euclidean distance computing similarity(semantically similar)
* [strong conclusions]Depth really is important for achieving results



## Notes

<!--(optional) 不在以上列表中，但需要特别记录的笔记。-->
what is *PCA* and *Data Argumentation* ?


## References

<!--(optional) 列出相关性高的文献，以便之后可以继续track下去。-->


---
# {{authorString}} ({{year}}) - {{title}}
[==*Read it now! See in Zotero*==]({{zoteroSelectURI}})
**Web:** [Open online]({{URL}})
**Citekey:** {{citekey}}
**Tags:** #source, (type), (status), (decade)
**PDF**: {{pdfAttachments}}


Type: {{entry.type}}
Authors: {{authorString}}
Year: {{year}}
Title: {{title}}
Journal: {{containerTitle}}
Pages: {{page}}
DOI: {{DOI}}
Publisher: {{publisher}}
Place: {{publisherPlace}}
---