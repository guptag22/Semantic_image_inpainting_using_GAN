# Image Inpainting with Semantic Context using Deep Generative Models

Garima Gupta
ggupta22@uic.edu

Shubham Singh
ssing57@uic.edu

Wangfei Wang
wwang75@uic.edu

Sai Teja Karnati
skarna3@uic.edu

### Abstract

Semantic inpainting is a task where a missing regions of images need to be filled based on the available visual data. It is very challenging especially when the missing regions are large. Here we show the implementation of the model proposed by Yeh et al. [1], where we first generated missing content by utilizing a trained deep convolutional generative adversarial network (DCGAN), and then searched the closest encoding of the corrupt image in the latent space using both the context loss and prior loss. The closest encoding was then passed to the generator used in DCGAN to generate the final inpainted results. We evaluated this model on three datasets, CelebA [2], DTD [3] and Pets [4]. While not showing equivalently goodresults as the original paper, we did find that this method generated relatively realistic images. We found that this method worked the best on the CelebA dataset and worked better with smaller missing region. We finally compared this method with another similar method Context Encoder (CE) [5], and found that CE tended to generate overly smooth images, but it performed pretty well according to our experiments.

### Introduction
Image inpainting refers to the task of filling in the missing parts of an image by inferring arbitrary large missing regions based on image semantics [1]. The task is challenging because missing pixels, in many cases, large missing regions, need to be inferred by the surrounding image pixels and high-leveled semantic context. Many methods were proposed to recover the missing regions. Classical methods are mostly based on single image inpainting that use either local or non-local information. For example, total variation based methods such as the method described by Afonso et al. [6] deconvolutes and reconstructs from compressive observations using total variation regularizations with the advantages of taking into account the smoothness of images. However this type of methods only works well with small missing holes. Many other methods were proposed for filling the missing holes by using similar information on the same image [7, 8, 9, 10]. However, these methods require similar pixels, structures or patches that are not flexible enough for real life applications. To tackle the task of inpainting for relatively large missing regions, non-local methods were also proposed by searching the external data. This kind of methods such as cutting and pasting semantically similar patch or fetching information from the internet[11, 12] have a major caveat that when test set is dramatically different from the training set, they would fail immediately. Recently, a semantic inpainting approach called Context Encoder (CE) was proposed by Pathak et al. [5], which is the closest to the method using in our project. Therefore, we will compare

1. Our method with CE in later Section 6. We will also describe briefly how CE works in Section
2. The major difference between the method proposed in Yeh et al. [1] and CE [5] is that deep convolutional generative adversarial network (DCGAN) used in Yeh et al. [1] does not need to train on the corrupted images.

We aim to implement the semantic inpainting method proposed by Yeh et. al [1] which utilized a trained DCGAN model, and then inferred the missing regions by conditioning on the available data. To inpaint the missing regions, we implemented the model where we searched the “closest” encoding of the corrupted image using the context and prior loss, and then we inferred the missing content by passing this encoding through the generative model in DCGAN [1].

Further deatiled description, explanation and results can be found in the project description pdf. 

Experimental results demonstrated that unlike the original papers [1], we failed to generate realistic images when the mask holes are large. We found that the method worked the best on the CelebA dataset. We compared the method on different mask shapes and sizes and found that the method worked very well on randomly placed small masks. CE generated overly smooth images. But in our experiments, the method worked pretty good. We acknowledge that with the recent development of this area, more advanced approaches may work better for semantic inpainting for large missing areas, such as contextual attention model proposed by Yu et al. [20]. To improve the performance of the DCGAN model, the future work may involve efficient hyperparameter tuning and testing on other large image datasets.

### References

[1] Raymond A. Yeh, Chen Chen, Teck Yian Lim, Alexander G. Schwing, Mark Hasegawa Johnson, and Minh N. Do. Semantic image inpainting with deep generative models. Proceedings - 30th IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2017, 2017-Janua:6882–6890, 2017. doi: 10.1109/CVPR.2017.728.

[2] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In Proceedings of International Conference on Computer Vision (ICCV), December 2015.

[3] M. Cimpoi, S. Maji, I. Kokkinos, S. Mohamed, , and A. Vedaldi. Describing textures in the wild. In Proceedings of the IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2014.

[4] O. M. Parkhi, A. Vedaldi, A. Zisserman, and C. V. Jawahar. Cats and dogs. In IEEE Conference on Computer Vision and Pattern Recognition, 2012.

[5] Deepak Pathak, Philipp Krahenbuhl, Jeff Donahue, Trevor Darrell, and Alexei A. Efros. Context Encoders: Feature Learning by Inpainting. Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 2016-Decem:2536–2544, 2016. ISSN 10636919. doi: 10.1109/CVPR.2016.278.

[6] Manya V. Afonso, Jose M. Bioucas-Dias, and Mrio A.T. Figueiredo. An augmented lagrangian approach to the constrained optimization formulation of imaging inverse problems. IEEE Transactions on Image Processing, 20(3):681–695, 2011. ISSN 10577149. doi: 10.1109/TIP.2010.2076294.

[7] Alexei A. Efros and Thomas K. Leung. Texture synthesis by non-parametric sampling. Proceedings of the IEEE International Conference on Computer Vision, 2(September):1033–1038, 1999. doi: 10.1109/iccv.1999.790383.

[8] Kaiming He and Jian Sun. Statistics of patch offsets for image completion. Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics), 7573 LNCS(PART 2):16–29, 2012. ISSN 03029743. doi: 10.1007/978-3-642-33709-3 2.

[9] Yao Hu, Debing Zhang, Jieping Ye, Xuelong Li, and Xiaofei He. Fast and accurate matrix completion via truncated nuclear norm regularization. IEEE Transactions on Pattern Analysis and Machine Intelligence, 35(9):2117–2130, 2013. ISSN 01628828. doi: 10.1109/TPAMI.2012.271.

[10] Jia Bin Huang, Sing Bing Kang, Narendra Ahuja, and Johannes Kopf. Image completion using planar structure guidance. ACM Transactions on Graphics, 33(4):1–10, 2014. ISSN 15577333.doi: 10.1145/2601097.2601205.

[11] Oliver Whyte, Josef Sivic, and Andrew Zisserman. Get out of my picture! Internet-based inpainting. British Machine Vision Conference, BMVC 2009 - Proceedings, 2009. doi: 10.5244/C.23.116.

[12] James Hays and Alexei A. Efros. Scene completion using millions of photographs. ACM Transactions on Graphics, 26(3):1–7, 2007. ISSN 07300301. doi: 10.1145/1276377.1276382.

[13] Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in Neural Information Processing Systems, 3(January):2672–2680, 2014. ISSN 10495258.

[14] Andrew Brock, Jeff Donahue, and Karen Simonyan. Large scale GaN training for high fidelity natural image synthesis. 7th International Conference on Learning Representations, ICLR 2019, pages 1–35, 2019.

[15] Phillip Isola, Jun Yan Zhu, Tinghui Zhou, and Alexei A. Efros. Image-to-image translation with conditional adversarial networks. Proceedings - 30th IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2017, 2017-January:5967–5976, 2017. doi: 10.1109/CVPR.2017.632.

[16] Guim Perarnau, Joost van de Weijer, Bogdan Raducanu, and Jose M. Alvarez. Invertible Conditional GANs for image editing. (Figure 1):1–9, 2016. URL http://arxiv.org/abs/1611.06355.

[17] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. ImageNet: A Large-Scale Hierarchical Image Database. In CVPR09, 2009.

[18] Alec Radford, Luke Metz, and Soumith Chintala. Unsupervised representation learning with deep convolutional generative adversarial networks. 4th International Conference on Learning Representations, ICLR 2016 - Conference Track Proceedings, pages 1–16, 2016.

[19] Laurens van der Maaten and Geoffrey Hinton. Visualizing Data using t-SNE. Journal of Machine Learning Research, 9:2579–2605, 2008.

[20] Jiahui Yu, Zhe Lin, Jimei Yang, Xiaohui Shen, Xin Lu, and Thomas S. Huang. Generative Image Inpainting with Contextual Attention. Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition, pages 5505–5514, 2018. ISSN 10636919.doi: 10.1109/CVPR.2018.00577.
