## 左边画图，右边解释
```latex
\begin{columns}
\column{.4\textwidth}
\begin{figure}
\centering
% Requires \usepackage{graphicx}
\includegraphics[width=6cm]{.jpg}
\end{figure}
\column{.5\textwidth}
文
\end{columns}
```

## 添加gif动画
gif转eps在线网站：[GIF转EPS - 免费在线将GIF文件转换成EPS (cdkm.com)](https://cdkm.com/cn/gif-to-eps)
ImageMagick下载网站：[ImageMagick – Download](https://www.imagemagick.org/script/download.php#windows)
具体步骤：
[Beamer中使用动态gif动画效果 - 简书 (jianshu.com)](https://www.jianshu.com/p/bf9859de1962)

## 插入表格
```latex
\documentclass{article}

\begin{document}

\begin{table}[h!]
  \begin{center}
    \caption{Your first table.}
    \begin{tabular}{l|c|r} % <-- Alignments: 1st column left, 2nd middle and 3rd right, with vertical lines in between
      \textbf{Value 1} & \textbf{Value 2} & \textbf{Value 3}\\
      $\alpha$ & $\beta$ & $\gamma$ \\
      \hline
      1 & 1110.1 & a\\
      2 & 10.1 & b\\
      3 & 23.113231 & c\\
    \end{tabular}
  \end{center}
\end{table}

\end{document}

```

## 排版多个图片
```latex
\usepackage{subfigure}
\usepackage{graphicx}
%包一定要加，不然会报错“error : Undefined control sequence”

\begin{figure*}[htbp]
\centering
 
\subfigure[]{
    \begin{minipage}[t]{0.33\linewidth}
        \centering
        \includegraphics[width=1.651in]{PR_Curve/FBMS_PR.eps}\\
        \vspace{0.02cm}
        \includegraphics[width=1.651in]{PR_Curve/DAVIS_PR.eps}\\
        \vspace{0.02cm}
        \includegraphics[width=1.651in]{PR_Curve/ViSal_PR.eps}\\
        \vspace{0.02cm}
        %\caption{fig1}
    \end{minipage}%
}%
\subfigure[]{
    \begin{minipage}[t]{0.33\linewidth}
        \centering
        \includegraphics[width=1.651in]{FMeasures/FBMS_Score.eps}\\
        \vspace{0.02cm}
        \includegraphics[width=1.651in]{FMeasures/DAVIS_Score.eps}\\
        \vspace{0.02cm}
        \includegraphics[width=1.651in]{FMeasures/ViSal_Score.eps}\\
        \vspace{0.02cm}
        %\caption{fig1}
    \end{minipage}%
}%
 
 
\centering
\caption{描述。。。}
\vspace{-0.2cm}
\label{fig:compare_fig}
\end{figure*}
```

## 调整字体
```latex


\small  
\bibliographystyle{ieee}  
\bibliography{CASSreference}

%Font Sizes  
%\tiny  
%\scriptsize  
%\footnotesize  
%\small  
%\normalsize  
%\large  
%\Large  
%\LARGE  
%\huge  
%\Huge
```

## 参考文献引用
```latex
\cite{}
%平齐引用
\textsuperscript{\cite{}}
%上标引用
```

## 调整图片位置
[(19条消息) 在 LaTeX 中调整图片和表格的位置_Xovee的博客-CSDN博客_latex表格位置](https://blog.csdn.net/xovee/article/details/109378160)
