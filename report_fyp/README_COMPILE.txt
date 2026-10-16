KickSense FYP Report - build and edit instructions
==================================================

WHAT IS IN THIS FOLDER
----------------------
main.tex              Preamble, title page, certificate, abstract, acknowledgement,
                      quote page, TOC, list of figures, list of tables, acronyms,
                      header setup, and the \input lines for each chapter.
ch1_introduction.tex  Chapter 1  - Introduction
ch2_literature.tex    Chapter 2  - Literature Review
ch3_methodology.tex   Chapter 3  - Methodology
ch4_dataset.tex       Chapter 4  - Dataset and Experimental Evaluation
ch5_conclusion.tex    Chapter 5  - Conclusion
ref.tex               References (thebibliography, 26 entries)
figures/              28 report figures plus BUI-Logo.jpg
main.pdf              The compiled 89 page report

The file layout, preamble, header style, table style and figure style all match
the sample report supplied by the supervisor.


HOW TO COMPILE ON OVERLEAF
--------------------------
1. Go to overleaf.com and choose New Project -> Upload Project.
2. Upload the zip that contains this folder's contents.
3. Open Menu (top left) and confirm:
     Compiler    = pdfLaTeX
     Main document = main.tex
4. Press Recompile. Press it a second time so the table of contents,
   list of figures and list of tables fill in.


HOW TO COMPILE LOCALLY (already done once)
------------------------------------------
From the repository root:

    .\tools\tectonic\tectonic.exe -X compile report_fyp\main.tex

Tectonic downloads whatever LaTeX packages it needs on first run, so the first
compile takes a few minutes and later ones take about twenty seconds.


HOW TO EDIT YOUR DETAILS
------------------------
Student names, registration numbers, supervisor and date are all in main.tex
in the titlepage block near the top, and again in the certificate block just
below it. Change them in both places.


HOW TO ADD OR REPLACE A FIGURE
------------------------------
Drop the image into figures/ and use the same pattern the report already uses:

    \begin{figure}[H]
        \centering
        \includegraphics[width=0.99\textwidth]{your_image.png}
        \caption{Your caption}
        \label{fig:your-label}
    \end{figure}

    \noindent
    \textbf{Description:} One paragraph explaining what the figure shows.

Every figure in this report is followed by a Description paragraph, which is
the convention used in the supervisor's sample. Keep it consistent.


HOW THE FIGURES AND NUMBERS WERE PRODUCED
-----------------------------------------
Nothing in this report was typed in by hand from memory. The scripts live in
report_assets/ at the repository root:

    make_figures.py     Generates all 28 figures and writes report_numbers.json
    run_ablation.py     Trains the six ablation configurations
    train_reference.py  Trains the seeded reference model used for every
                        test-set number, confusion matrix and per-class metric

To regenerate everything:

    .\kicksense_env\Scripts\python.exe report_assets\run_ablation.py
    .\kicksense_env\Scripts\python.exe report_assets\train_reference.py
    .\kicksense_env\Scripts\python.exe report_assets\make_figures.py
