Development of a Python-Based Automated History
Matching Tool for Reservoir Performance Analysis and

Forecasting

Project topic: Development of a Python-Based Automated History Matching Tool for Reservoir
Performance Analysis and Forecasting — OPM Flow + EnKF + Web Interface
Student: Osimi Godsgift Gbubemi
Registration No.: 20211276493
Supervisor: Dr Chikwe
Department: Petroleum Engineering
Project Type: Final Year Project — Individual
High-level Stack: OPM Flow, Python (Django), EnKF, React (Vite), PostgreSQL, Render
Abstract: This project develops an accessible, educational, and deployable platform that automates
reservoir history matching to improve reservoir performance analysis and forecasting. The tool integrates
the open-source simulator OPM Flow with automated calibration algorithms — notably the Ensemble
Kalman Filter (EnKF) — and a baseline manual matching option. The system will produce predictions that
align reservoir models with real production data, quantify prediction uncertainty, and provide a web
interface for interactive visualization and teaching.
Problem & Importance: Translating field measurements (pressures and production rates) into reliable
reservoir forecasts is traditionally manual, time-consuming, and often inaccessible to students due to
licensing and complexity. Inaccurate models produce misleading forecasts that can result in poor field
decisions. This project focuses on building a reproducible, low-cost tool that automates matching against
real production data, delivers predictive forecasts, and serves as an educational platform enabling
lower-level students to gain hands-on experience in reservoir modeling and decision-making.
Aims & Objectives:
Main Aim: Build and demonstrate a Python-based automated history matching tool that aligns reservoir
model predictions with real production data for improved performance analysis and forecasting.
Objectives:
1. Implement a baseline manual matching interface to illustrate traditional parameter tuning methods and
their limitations.
2. Implement an Ensemble Kalman Filter (EnKF) workflow to automate parameter estimation and provide
uncertainty quantification.
3. Integrate OPM Flow with a FastAPI backend to execute forward simulations against observed (real or
synthetic) production data.
4. Produce forecast runs that compare prior and posterior model predictions, highlighting improvements in
predictive accuracy.
5. Develop a simple React frontend for interactive visualization aimed at educational use, and deploy the
demo on Render for open access by students and lecturers.
Methodology & Tools:
• Data: Start with synthetic and publicly available benchmark cases; extend to small real-field or
anonymized datasets if available to demonstrate prediction against real situations.
• Simulator: OPM Flow invoked via Python bindings for forward model simulations.
• Algorithms: Baseline manual matching and Ensemble Kalman Filter (EnKF) for automated calibration and
uncertainty quantification.
• Backend/API: FastAPI to orchestrate jobs, manage ensembles, and serve results to the frontend.

• Frontend: React (Vite) with Recharts or Plotly for interactive visualizations (observed vs simulated,
ensemble spread, parameter histograms, and forecast comparisons).
• Database & Deployment: PostgreSQL (Render) for metadata and result storage; deploy web service on
Render for accessible teaching demos.
Expected Outcomes & Deliverables:
• A web-accessible demo that performs baseline and EnKF-based history matching, produces forecasts,
and visualizes results (observed vs simulated, ensemble spread, parameter posteriors, and forecast
comparison).
• Source code repository with Docker configuration, reproducible instructions, and a concise student guide
for hands-on exercises.
• Project report and a short recorded demo as a backup for presentation day.
• Deployed demo on Render with access instructions enabling lecturers and students to explore and learn
interactively.
High-level Timeline (6 weeks): Week 1 — dataset & simulator validation; Weeks 2–3 — backend &
baseline matching; Week 4 — EnKF implementation and tests; Week 5 — frontend integration; Week 6 —
deployment, documentation, and rehearsal.
Prepared by: Osimi Godsgift Gbubemi
Registration No.: 20211276493
Supervisor: Dr Chikwe

FRONTEND PAGES.

HOME PAGE:
A MODERN CLASSIC BREATH TAKNG HOME PAGE THAT SHOWS USING WRITEUP CONTENTS AND IMAGES SHORT EXPLANATION ON WHAT RESERVIOUR SIMULATION MEANS CONCEPT BY CONCEPT WITH A SEE MORE LINK AT THE END OF EACH WRITE UP LINKING THE USER TO THE 
CONCEPT PAGE.

CONCEPT PAGE:
THIS PAGE SHOWS A PAGINATED LIST OF THE VARIOUS CONCEPT ABOUT RESEEVOIR SIMULATION AND HISTORY MATCHING WITH THE OPTION TO VIEW ENTIRE DETAILS WHICH WHEN CLICKED OPENS UP THE FULL DETAILS OF THAT CONCEPT

USER GUILD PAGE:
THIS PAGE INSTRUCT AND GUILDS THE USER ON HOW TO USE THE PLATFORM TO PERFORM HISTORY MATCHING AND SIMULATION

XCAPE SIMULATOR:
This is the actual page where the user actually get to use the services, it should contain all neccessary user interface and should be highly graphical with profesional styling with field oriented styling.

ABOUT PAGE:
This page contains all details about the project, problem statement and what the entire project trys to solve. 

PROFILES AND CONTRIBUTORS
This section shows the detail of all contributors, my supervisor profile, my profile and any other profile. The profile should show
-image
-qualification
-bio
-Achievement
-Role
-Contact
-address
etc

CONTACT PAGE:
UNIVERSITY, SUPERVISOR, MY PROFILE ETC
 PLATFORM COLOR BREAKDOWN
Below is the breakdown of the color combos i intend using.
   Primary: #0F4C81 — deep blue
Secondary: #1F7A8C — teal
Accent: #F4B400 — amber/gold
Background: #F7F9FC — very light gray-blue
Card surface: #FFFFFF
Text: #1F2937 — dark slate
Success: #22C55E
Warning: #F59E0B
Error: #EF4444


KEY STACK AND IMPORTANT NOTE.
USE MATERIAL UI
PROPERY USE OF REACT HOOK AND REDUX REACT
CHAD CN
MODERN FRONTEND TOOLS AND OPTIMIZER
IT SHOULD HAVE MODE TOGGLER TO TOGGLE BETWEEN DARK MODE AND LIGHT MODE 
PATH TO FRONTEND FOLDER PS C:\Users\USER\Desktop\XCAPE\frontend>
path to backend folder
PS C:\Users\USER\Desktop\XCAPE\backend\XCAPE>
PATH TO LOGOS
WHITE BG C:\Users\USER\Desktop\XCAPE\frontend\Xcape.png
BLACK  BG frontend\xcapeblackBG.png


Note XCAPE IS THE NAME OF THE PROJECT PLATFORM SO THIS SHOULD BE IN THE HOME PAGE WITH GOOD LOGO STYLING

IMAGE FOLDER
the image folder contains images i want you to use when designing the frontend here is the path to the folder C:\Users\USER\Desktop\XCAPE\frontend\images

This document gives a complete flow on what we intend to build and to achieve.
PHASES BREAK DOWN

