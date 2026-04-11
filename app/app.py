from pathlib import Path
from textwrap import dedent

import streamlit as st
import streamlit.components.v1 as components

from core.top_nav import hide_default_multipage_nav, render_flow_nav


def _render_home_journey_section() -> None:
    st.subheader("\u2728 Zobacz, jak p\u0142yn\u0105 Twoje dane")
    st.caption(
        "Najkr\u00f3tsza droga prowadzi z Automatu EDA prosto do modelu. "
        "Data Chat mo\u017cesz pomin\u0105\u0107 albo potraktowa\u0107 jak osobne centrum analityczne do deep dive."
    )

    html = dedent(
        """
        <div class="ae-story">
          <style>
            :root{
              --ae-red:#ff4b4b;
              --ae-blue:#60a5fa;
              --ae-blue-dark:#2563eb;
              --ae-gold:#c58b00;
              --ae-gold-soft:#fff6d7;
              --ae-ink:#1f2937;
              --ae-muted:#64748b;
              --ae-border:#e5e7eb;
              --ae-panel:#f8fbff;
              --ae-shadow:0 20px 50px rgba(15,23,42,.08);
            }

            .ae-story{
              font-family:"Segoe UI", Arial, sans-serif;
              color:var(--ae-ink);
              padding:0.15rem 0 0.5rem 0;
            }

            .ae-board{
              position:relative;
              overflow:hidden;
              border:1px solid #e6edf7;
              border-radius:30px;
              background:#ffffff;
              box-shadow:var(--ae-shadow);
              padding:24px 22px 28px 22px;
            }

            .ae-topbar{
              position:relative;
              z-index:3;
              display:flex;
              flex-wrap:wrap;
              gap:10px;
              margin-bottom:14px;
            }

            .ae-pill{
              display:inline-flex;
              align-items:center;
              gap:8px;
              padding:8px 12px;
              border-radius:999px;
              border:1px solid #d8e4ff;
              background:#eff6ff;
              color:#1d4ed8;
              font-size:13px;
              font-weight:700;
              box-shadow:0 10px 18px rgba(37,99,235,.08);
            }

            .ae-pill.alt{
              border-color:#ffd0d0;
              background:#fff3f3;
              color:#b91c1c;
            }

            .ae-pill .dot{
              width:10px;
              height:10px;
              border-radius:50%;
              background:var(--ae-blue-dark);
              box-shadow:0 0 0 0 rgba(37,99,235,.35);
              animation:aePulseBlue 2.4s ease-in-out infinite;
            }

            .ae-pill.alt .dot{
              background:var(--ae-red);
              box-shadow:0 0 0 0 rgba(255,75,75,.28);
              animation:aePulseRed 2.4s ease-in-out infinite;
            }

            .ae-hero-copy{
              position:relative;
              z-index:3;
              margin-bottom:20px;
              padding:18px 20px;
              border-radius:22px;
              background:rgba(255,255,255,.80);
              border:1px solid #e9eef7;
              backdrop-filter:blur(8px);
            }

            .ae-hero-copy strong{
              display:block;
              margin-bottom:8px;
              font-size:16px;
              color:#0f172a;
            }

            .ae-hero-copy span{
              color:#475569;
              font-size:14px;
              line-height:1.6;
            }

            .ae-main-row{
              position:relative;
              z-index:3;
              display:grid;
              grid-template-columns:minmax(0,1fr) 68px minmax(0,1fr) 68px minmax(0,1fr) 68px minmax(0,1fr);
              align-items:stretch;
              gap:0;
            }

            .ae-connector{
              position:relative;
              display:flex;
              align-items:center;
              justify-content:center;
              min-height:270px;
            }

            .ae-connector-line{
              position:absolute;
              left:6px;
              right:20px;
              top:50%;
              height:4px;
              border-radius:999px;
              transform:translateY(-50%);
              background:linear-gradient(90deg, rgba(147,197,253,.28), rgba(255,75,75,.30), rgba(147,197,253,.28));
              overflow:hidden;
            }

            .ae-connector-line::before{
              content:"";
              position:absolute;
              left:-18%;
              top:50%;
              width:30px;
              height:30px;
              border-radius:50%;
              transform:translateY(-50%);
              background:radial-gradient(circle at 30% 30%, #ffffff 0 18%, #93c5fd 30%, #2563eb 62%, #1d4ed8 100%);
              box-shadow:0 0 0 8px rgba(96,165,250,.12), 0 0 24px rgba(37,99,235,.34);
              animation:aeTravel 4.4s linear infinite;
            }

            .ae-connector-line::after{
              content:"";
              position:absolute;
              left:-35%;
              top:50%;
              width:110px;
              height:8px;
              transform:translateY(-50%);
              background:linear-gradient(90deg, rgba(96,165,250,0), rgba(96,165,250,.42), rgba(255,75,75,.22), rgba(96,165,250,0));
              filter:blur(4px);
              animation:aeBeam 4.4s linear infinite;
            }

            .ae-connector.delay-2 .ae-connector-line::before,
            .ae-connector.delay-2 .ae-connector-line::after{
              animation-delay:1.1s;
            }

            .ae-connector.delay-3 .ae-connector-line::before,
            .ae-connector.delay-3 .ae-connector-line::after{
              animation-delay:2.1s;
            }

            .ae-arrow{
              position:absolute;
              right:0;
              top:50%;
              transform:translateY(-50%);
              z-index:2;
              width:24px;
              height:24px;
              display:flex;
              align-items:center;
              justify-content:center;
            }

            .ae-arrow svg{
              display:block;
              width:24px;
              height:24px;
              overflow:visible;
            }

            .ae-arrow polyline{
              stroke:#7b8798;
              stroke-width:4;
              stroke-linecap:round;
              stroke-linejoin:round;
              fill:none;
            }

            .ae-node{
              position:relative;
              min-height:248px;
              padding:16px 16px 72px 16px;
              border-radius:24px;
              border:1px solid var(--ae-border);
              background:rgba(255,255,255,.94);
              box-shadow:0 18px 40px rgba(15,23,42,.08);
              animation:aeFloat 7s ease-in-out infinite;
            }

            .ae-node.stage-1{animation-delay:0s;}
            .ae-node.stage-2{animation-delay:.35s;}
            .ae-node.stage-4{animation-delay:.7s;}
            .ae-node.stage-5{animation-delay:1.05s;}

            .ae-head{
              display:flex;
              align-items:center;
              gap:10px;
              margin-bottom:10px;
            }

            .ae-step{
              display:inline-flex;
              align-items:center;
              justify-content:center;
              width:30px;
              height:30px;
              border-radius:10px;
              background:linear-gradient(180deg, #79b2ff 0%, #4f8ff2 100%);
              color:#fff;
              font-size:15px;
              font-weight:800;
              box-shadow:0 8px 20px rgba(79,143,242,.28);
            }

            .ae-title{
              margin:0;
              font-size:17px;
              font-weight:800;
              color:var(--ae-ink);
            }

            .ae-value{
              position:absolute;
              top:14px;
              right:14px;
              padding:6px 10px;
              border-radius:999px;
              background:#fff3f3;
              color:#b91c1c;
              font-size:11px;
              font-weight:800;
              letter-spacing:.02em;
              text-transform:uppercase;
            }

            .ae-copy{
              margin:0 0 12px 0;
              color:var(--ae-muted);
              font-size:13px;
              line-height:1.55;
            }

            .ae-list{
              display:grid;
              gap:8px;
            }

            .ae-chip{
              display:flex;
              align-items:center;
              gap:8px;
              padding:9px 10px;
              border-radius:14px;
              background:var(--ae-panel);
              border:1px solid #e4ecf7;
              color:#334155;
              font-size:12px;
              font-weight:700;
              opacity:0;
              transform:translateY(10px);
              animation:aeReveal 8.8s ease-in-out infinite;
            }

            .ae-chip::before{
              content:"";
              width:8px;
              height:8px;
              border-radius:50%;
              background:var(--ae-red);
              flex:0 0 8px;
              box-shadow:0 0 0 0 rgba(255,75,75,.18);
              animation:aePulseRed 2.2s ease-in-out infinite;
            }

            .ae-output{
              position:absolute;
              left:14px;
              right:14px;
              bottom:14px;
              display:flex;
              flex-wrap:wrap;
              gap:8px;
            }

            .ae-output-chip{
              display:inline-flex;
              align-items:center;
              gap:7px;
              padding:7px 10px;
              border-radius:999px;
              border:1px solid #dbe6f6;
              background:rgba(255,255,255,.94);
              color:#334155;
              font-size:11px;
              font-weight:800;
              box-shadow:0 12px 18px rgba(15,23,42,.07);
              opacity:0;
              transform:translateY(18px);
              animation:aeOutputPop 8.8s ease-in-out infinite;
            }

            .ae-output-chip::before{
              content:"";
              width:8px;
              height:8px;
              border-radius:50%;
              background:#1d4ed8;
              flex:0 0 8px;
              box-shadow:0 0 0 0 rgba(37,99,235,.28);
              animation:aePulseBlue 2.4s ease-in-out infinite;
            }

            .ae-node.stage-1 .ae-chip:nth-child(1){animation-delay:.10s;}
            .ae-node.stage-1 .ae-chip:nth-child(2){animation-delay:.35s;}
            .ae-node.stage-1 .ae-chip:nth-child(3){animation-delay:.60s;}
            .ae-node.stage-1 .ae-output-chip:nth-child(1){animation-delay:.82s;}
            .ae-node.stage-1 .ae-output-chip:nth-child(2){animation-delay:1.02s;}

            .ae-node.stage-2 .ae-chip:nth-child(1){animation-delay:1.10s;}
            .ae-node.stage-2 .ae-chip:nth-child(2){animation-delay:1.35s;}
            .ae-node.stage-2 .ae-chip:nth-child(3){animation-delay:1.60s;}
            .ae-node.stage-2 .ae-output-chip:nth-child(1){animation-delay:1.84s;}
            .ae-node.stage-2 .ae-output-chip:nth-child(2){animation-delay:2.06s;}

            .ae-node.stage-4 .ae-chip:nth-child(1){animation-delay:2.20s;}
            .ae-node.stage-4 .ae-chip:nth-child(2){animation-delay:2.45s;}
            .ae-node.stage-4 .ae-chip:nth-child(3){animation-delay:2.70s;}
            .ae-node.stage-4 .ae-output-chip:nth-child(1){animation-delay:2.95s;}
            .ae-node.stage-4 .ae-output-chip:nth-child(2){animation-delay:3.15s;}

            .ae-node.stage-5 .ae-chip:nth-child(1){animation-delay:3.25s;}
            .ae-node.stage-5 .ae-chip:nth-child(2){animation-delay:3.50s;}
            .ae-node.stage-5 .ae-chip:nth-child(3){animation-delay:3.75s;}
            .ae-node.stage-5 .ae-output-chip:nth-child(1){animation-delay:4.00s;}
            .ae-node.stage-5 .ae-output-chip:nth-child(2){animation-delay:4.20s;}

            .ae-optional-wrap{
              position:relative;
              z-index:3;
              margin-top:30px;
              padding-top:120px;
            }

            .ae-optional-svg{
              position:absolute;
              top:0;
              left:0;
              width:100%;
              height:164px;
              overflow:visible;
              pointer-events:none;
            }

            .ae-branch-path{
              fill:none;
              stroke:rgba(96,165,250,.72);
              stroke-width:3;
              stroke-dasharray:7 10;
              stroke-linecap:round;
              stroke-linejoin:round;
              animation:aeBranch 4.6s ease-in-out infinite;
            }

            .ae-branch-dot-html{
              position:absolute;
              top:0;
              left:0;
              width:22px;
              height:22px;
              border-radius:50%;
              pointer-events:none;
              z-index:2;
              background:radial-gradient(circle at 30% 30%, #ffffff 0 18%, #93c5fd 30%, #2563eb 62%, #1d4ed8 100%);
              box-shadow:0 0 0 6px rgba(96,165,250,.12), 0 0 20px rgba(37,99,235,.28);
              will-change:transform;
            }

            .ae-optional-node{
              position:relative;
              z-index:4;
              max-width:520px;
              margin:0 auto;
              padding:18px 18px 78px 18px;
              border-radius:26px;
              border:1px solid transparent;
              background:
                linear-gradient(180deg, rgba(255,255,255,.98), rgba(248,251,255,.98)) padding-box,
                linear-gradient(135deg, rgba(96,165,250,.82), rgba(197,139,0,.40)) border-box;
              box-shadow:0 24px 52px rgba(37,99,235,.14);
              animation:aeFloat 6.4s ease-in-out infinite;
              animation-delay:.5s;
            }

            .ae-optional-node::before{
              content:"";
              position:absolute;
              inset:-20px;
              border-radius:36px;
              background:radial-gradient(circle, rgba(96,165,250,.18), transparent 62%);
              filter:blur(18px);
              z-index:-1;
            }

            .ae-badge{
              display:inline-flex;
              align-items:center;
              gap:8px;
              margin-bottom:12px;
              padding:7px 12px;
              border-radius:999px;
              background:#fff1f2;
              color:#b91c1c;
              border:1px solid #fecdd3;
              font-size:12px;
              font-weight:800;
              letter-spacing:.01em;
              box-shadow:0 12px 20px rgba(255,75,75,.10);
            }

            .ae-badge::before{
              content:"";
              width:9px;
              height:9px;
              border-radius:50%;
              background:var(--ae-red);
              box-shadow:0 0 0 0 rgba(255,75,75,.32);
              animation:aePulseRed 2.2s ease-in-out infinite;
            }

            .ae-optional-node .ae-value{
              background:var(--ae-gold-soft);
              color:var(--ae-gold);
              border:1px solid #f2df9e;
            }

            .ae-optional-node .ae-chip:nth-child(1){animation-delay:4.15s;}
            .ae-optional-node .ae-chip:nth-child(2){animation-delay:4.40s;}
            .ae-optional-node .ae-chip:nth-child(3){animation-delay:4.65s;}
            .ae-optional-node .ae-output-chip:nth-child(1){animation-delay:4.92s;}
            .ae-optional-node .ae-output-chip:nth-child(2){animation-delay:5.14s;}

            .ae-outcomes{
              position:relative;
              z-index:3;
              display:grid;
              grid-template-columns:repeat(3, minmax(0,1fr));
              gap:12px;
              margin-top:26px;
            }

            .ae-outcome{
              padding:15px 15px 14px 15px;
              border-radius:18px;
              background:rgba(255,255,255,.88);
              border:1px solid #e5ecf6;
            }

            .ae-outcome h4{
              margin:0 0 7px 0;
              font-size:13px;
              font-weight:800;
              color:#0f172a;
            }

            .ae-outcome p{
              margin:0;
              font-size:13px;
              line-height:1.55;
              color:#475569;
            }

            @keyframes aeFloat{
              0%,100%{transform:translateY(0);}
              50%{transform:translateY(-6px);}
            }

            @keyframes aePulseBlue{
              0%{box-shadow:0 0 0 0 rgba(37,99,235,.28);}
              70%{box-shadow:0 0 0 12px rgba(37,99,235,0);}
              100%{box-shadow:0 0 0 0 rgba(37,99,235,0);}
            }

            @keyframes aePulseRed{
              0%{box-shadow:0 0 0 0 rgba(255,75,75,.22);}
              70%{box-shadow:0 0 0 12px rgba(255,75,75,0);}
              100%{box-shadow:0 0 0 0 rgba(255,75,75,0);}
            }

            @keyframes aeReveal{
              0%,8%{opacity:0; transform:translateY(10px);}
              12%,26%{opacity:1; transform:translateY(0);}
              34%,100%{opacity:.9; transform:translateY(0);}
            }

            @keyframes aeOutputPop{
              0%,14%{opacity:0; transform:translateY(18px) scale(.96);}
              18%,32%{opacity:1; transform:translateY(0) scale(1);}
              40%,100%{opacity:.95; transform:translateY(0) scale(1);}
            }

            @keyframes aeTravel{
              0%{left:-18%;}
              100%{left:100%;}
            }

            @keyframes aeBeam{
              0%{left:-35%;}
              100%{left:100%;}
            }

            @keyframes aeBranch{
              0%,100%{opacity:.62;}
              50%{opacity:1;}
            }

            @media (max-width: 1180px){
              .ae-main-row{
                grid-template-columns:1fr;
                gap:12px;
              }

              .ae-connector{
                min-height:54px;
              }

              .ae-arrow{
                width:40px;
                height:40px;
              }

              .ae-optional-wrap{
                padding-top:18px;
              }

              .ae-optional-svg{
                display:none;
              }

              .ae-outcomes{
                grid-template-columns:1fr;
              }
            }
          </style>

          <div class="ae-board">
            <div class="ae-topbar">
              <div class="ae-pill"><span class="dot"></span>G&#322;&oacute;wny tor: 1 &rarr; 2 &rarr; 4 &rarr; 5</div>
              <div class="ae-pill alt"><span class="dot"></span>Etap 3 to opcjonalny deep dive, kt&oacute;ry mo&#380;esz potraktowa&#263; jak osobn&#261; aplikacj&#281;</div>
            </div>

            <div class="ae-hero-copy">
              <strong>Od surowego pliku do modelu i predykcji bez przepinania danych.</strong>
              <span>
                Najpierw porz&#261;dkujesz i diagnozujesz dane, potem przechodzisz prosto do modelowania,
                a je&#347;li chcesz wej&#347;&#263; g&#322;&#281;biej, odbijasz do Data Chat po dodatkowe insighty,
                interpretacje i executive takeaways.
              </span>
            </div>

            <div class="ae-main-row">
              <div class="ae-node stage-1">
                <span class="ae-value">fundament</span>
                <div class="ae-head">
                  <span class="ae-step">1</span>
                  <h3 class="ae-title">Analiza Danych</h3>
                </div>
                <p class="ae-copy">Startujesz od pliku albo demo i od razu ustawiasz solidny fundament pod dalsz&#261; prac&#281;.</p>
                <div class="ae-list">
                  <div class="ae-chip">upload, podgl&#261;d i sanity check</div>
                  <div class="ae-chip">rozpoznanie typu zadania i roli kolumn</div>
                  <div class="ae-chip">artefakty zapisane dla kolejnych etap&oacute;w</div>
                </div>
                <div class="ae-output">
                  <div class="ae-output-chip">profil danych</div>
                  <div class="ae-output-chip">gotowy punkt startu</div>
                </div>
              </div>

              <div class="ae-connector delay-1">
                <div class="ae-connector-line"></div>
                <div class="ae-arrow" aria-hidden="true">
                  <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <polyline points="4,4 18,12 4,20"></polyline>
                  </svg>
                </div>
              </div>

              <div class="ae-node stage-2">
                <span class="ae-value">decyzje</span>
                <div class="ae-head">
                  <span class="ae-step">2</span>
                  <h3 class="ae-title">Automat EDA</h3>
                </div>
                <p class="ae-copy">Dostajesz szybki audyt jako&#347;ci, cleaning i gotowe rekomendacje bez r&#281;cznego przeklikiwania raport&oacute;w.</p>
                <div class="ae-list">
                  <div class="ae-chip">braki, duplikaty, outliery i ryzyka</div>
                  <div class="ae-chip">automatyczne czyszczenie i feature prep</div>
                  <div class="ae-chip">TL;DR i wskaz&oacute;wki do kolejnego kroku</div>
                </div>
                <div class="ae-output">
                  <div class="ae-output-chip">ready_for_training</div>
                  <div class="ae-output-chip">rekomendacje</div>
                </div>
              </div>

              <div class="ae-connector delay-2">
                <div class="ae-connector-line"></div>
                <div class="ae-arrow" aria-hidden="true">
                  <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <polyline points="4,4 18,12 4,20"></polyline>
                  </svg>
                </div>
              </div>

              <div class="ae-node stage-4">
                <span class="ae-value">model</span>
                <div class="ae-head">
                  <span class="ae-step">4</span>
                  <h3 class="ae-title">Trenowanie modelu</h3>
                </div>
                <p class="ae-copy">Gdy dane s&#261; gotowe, uruchamiasz modelowanie szybciej, bo pipeline ju&#380; niesie przygotowane artefakty.</p>
                <div class="ae-list">
                  <div class="ae-chip">AutoML i por&oacute;wnanie modeli</div>
                  <div class="ae-chip">tuning, walidacja i wyb&oacute;r zwyci&#281;zcy</div>
                  <div class="ae-chip">kr&oacute;tka droga od danych do modelu</div>
                </div>
                <div class="ae-output">
                  <div class="ae-output-chip">ranking modeli</div>
                  <div class="ae-output-chip">zwyci&#281;zca pipeline</div>
                </div>
              </div>

              <div class="ae-connector delay-3">
                <div class="ae-connector-line"></div>
                <div class="ae-arrow" aria-hidden="true">
                  <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <polyline points="4,4 18,12 4,20"></polyline>
                  </svg>
                </div>
              </div>

              <div class="ae-node stage-5">
                <span class="ae-value">wynik</span>
                <div class="ae-head">
                  <span class="ae-step">5</span>
                  <h3 class="ae-title">Predykcja</h3>
                </div>
                <p class="ae-copy">Ko&#324;czysz praktycznym wynikiem: predykcj&#261; punktow&#261; albo batchow&#261; gotow&#261; do pobrania i u&#380;ycia.</p>
                <div class="ae-list">
                  <div class="ae-chip">scoring jednego rekordu lub ca&#322;ej paczki</div>
                  <div class="ae-chip">walidacja wej&#347;cia i eksport wynik&oacute;w</div>
                  <div class="ae-chip">ostatni krok gotowy dla biznesu</div>
                </div>
                <div class="ae-output">
                  <div class="ae-output-chip">CSV / XLSX wynik&oacute;w</div>
                  <div class="ae-output-chip">scoring gotowy do wdro&#380;enia</div>
                </div>
              </div>
            </div>

            <div class="ae-optional-wrap">
              <svg class="ae-optional-svg" viewBox="0 0 1000 164" preserveAspectRatio="none" aria-hidden="true">
                <path id="aeBranchPathLeft" class="ae-branch-path" d="M 300 0 L 300 116 Q 300 142 326 142 L 500 142"></path>
                <path id="aeBranchPathRight" class="ae-branch-path" d="M 500 142 L 674 142 Q 700 142 700 116 L 700 0"></path>
              </svg>
              <div class="ae-branch-dot-html ae-branch-dot-left" aria-hidden="true"></div>
              <div class="ae-branch-dot-html ae-branch-dot-right" aria-hidden="true"></div>
              <div class="ae-optional-node">
                <span class="ae-value">INSIGHTY PREMIUM</span>
                <div class="ae-badge">Opcjonalny przystanek o du&#380;ej warto&#347;ci</div>
                <div class="ae-head">
                  <span class="ae-step">3</span>
                  <h3 class="ae-title">Data Chat</h3>
                </div>
                <p class="ae-copy">Je&#347;li chcesz zrozumie&#263; dane g&#322;&#281;biej, mo&#380;esz odbi&#263; do etapu 3 i potraktowa&#263; go jak osobne centrum discovery oraz insight&oacute;w.</p>
                <div class="ae-list">
                  <div class="ae-chip">pytania po polsku do danych i wykres&oacute;w</div>
                  <div class="ae-chip">interpretacje, executive takeaways i deep dive</div>
                  <div class="ae-chip">mocny dodatek, ale nie blokuje &#347;cie&#380;ki do modelu</div>
                </div>
                <div class="ae-output">
                  <div class="ae-output-chip">wykresy i odpowiedzi</div>
                  <div class="ae-output-chip">narracja dla decydenta</div>
                </div>
              </div>
            </div>

            <div class="ae-outcomes">
              <div class="ae-outcome">
                <h4>Co robi aplikacja</h4>
                <p>Przenosi dane i artefakty przez kolejne etapy, wi&#281;c nie zaczynasz od zera na ka&#380;dej stronie.</p>
              </div>
              <div class="ae-outcome">
                <h4>Co dostaje u&#380;ytkownik</h4>
                <p>Od sanity checku i EDA, przez opcjonalne insighty, a&#380; po model i gotowe predykcje.</p>
              </div>
              <div class="ae-outcome">
                <h4>Jak czyta&#263; ten przep&#322;yw</h4>
                <p>Najkr&oacute;tsza droga to 1 &rarr; 2 &rarr; 4 &rarr; 5, a etap 3 jest &#347;wiadomym wyborem dla bogatszej analizy.</p>
              </div>
            </div>
          </div>
        </div>
        <script>
          (() => {
            const svg = document.querySelector('.ae-optional-svg');
            const leftPath = document.getElementById('aeBranchPathLeft');
            const rightPath = document.getElementById('aeBranchPathRight');
            const leftDot = document.querySelector('.ae-branch-dot-left');
            const rightDot = document.querySelector('.ae-branch-dot-right');
            if (!svg || !leftPath || !rightPath || !leftDot || !rightDot) return;

            const duration = 5400;
            const secondDelay = 2700;
            let rafId = null;

            const placeDot = (dot, path, progress) => {
              const length = path.getTotalLength();
              const point = path.getPointAtLength(length * Math.max(0, Math.min(1, progress)));
              const rect = svg.getBoundingClientRect();
              const viewBox = svg.viewBox.baseVal;
              const scaleX = viewBox && viewBox.width ? rect.width / viewBox.width : 1;
              const scaleY = viewBox && viewBox.height ? rect.height / viewBox.height : 1;
              const x = point.x * scaleX;
              const y = point.y * scaleY;
              dot.style.transform = `translate(${x - 11}px, ${y - 11}px)`;
            };

            const tick = (ts) => {
              const p1 = (ts % duration) / duration;
              const p2 = (((ts - secondDelay) % duration) + duration) % duration / duration;
              placeDot(leftDot, leftPath, p1);
              placeDot(rightDot, rightPath, p2);
              rafId = window.requestAnimationFrame(tick);
            };

            rafId = window.requestAnimationFrame(tick);
            window.addEventListener('beforeunload', () => {
              if (rafId !== null) window.cancelAnimationFrame(rafId);
            });
          })();
        </script>
        """
    )

    components.html(html, height=1260, scrolling=False)


def _render_datachat_question_router_section() -> None:
    st.markdown("### 🧭 Jak pytanie uruchamia właściwą gałąź w Data Chat")
    st.caption(
        "Najprościej: najpierw wybierasz rodzaj pytania, a dopiero potem Data Chat dobiera "
        "gałąź odpowiedzi i zestaw wykresów najlepiej dopasowany do celu pytania."
    )

    chart_path = Path(__file__).resolve().parent / "assets" / "andrew_abela_chart_chooser.jpg"
    st.image(
        str(chart_path),
        caption="Schemat referencyjny: Andrew Abela — typ pytania determinuje najlepszą rodzinę wizualizacji.",
        width="stretch",
    )

    st.markdown(
        dedent(
            """
            <style>
              .dc-router-list {
                display: grid;
                gap: 14px;
                margin-top: 0.75rem;
              }

              .dc-router-card {
                background: #ffffff;
                border: 1px solid rgba(127, 157, 215, 0.20);
                border-radius: 18px;
                padding: 16px 18px;
                box-shadow: 0 10px 24px rgba(79, 99, 143, 0.06);
              }

              .dc-router-head {
                display: flex;
                align-items: center;
                gap: 10px;
                margin-bottom: 12px;
                font-weight: 800;
                color: #1f2b45;
                font-size: 16px;
              }

              .dc-router-dot {
                width: 11px;
                height: 11px;
                border-radius: 50%;
                background: #5f98ff;
                box-shadow: 0 0 0 5px rgba(95, 152, 255, 0.10);
                flex: 0 0 auto;
              }

              .dc-router-grid {
                display: grid;
                grid-template-columns: 1.35fr 1.5fr 1.1fr;
                gap: 12px;
              }

              .dc-router-block {
                background: linear-gradient(180deg, #fbfdff 0%, #f3f7ff 100%);
                border: 1px solid rgba(127, 157, 215, 0.14);
                border-radius: 14px;
                padding: 12px 13px;
              }

              .dc-router-label {
                display: inline-block;
                margin-bottom: 8px;
                font-size: 11px;
                font-weight: 800;
                letter-spacing: 0.04em;
                text-transform: uppercase;
                color: #5f6f91;
              }

              .dc-router-question {
                color: #243454;
                font-weight: 700;
                line-height: 1.55;
              }

              .dc-router-trigger {
                color: #4d5d7d;
                line-height: 1.62;
              }

              .dc-router-output {
                color: #3e4c67;
                line-height: 1.65;
              }

              .dc-router-note {
                margin-top: 0.95rem;
                padding: 14px 16px;
                border-radius: 16px;
                background: rgba(95, 152, 255, 0.09);
                border: 1px solid rgba(95, 152, 255, 0.16);
                color: #33415f;
                font-size: 14px;
                line-height: 1.6;
              }

              @media (max-width: 980px) {
                .dc-router-grid {
                  grid-template-columns: 1fr;
                }
              }
            </style>
            """
        ),
        unsafe_allow_html=True,
    )

    rows = [
        {
            "branch": "Distribution",
            "question": "„Pokaż rozkład wartości (histogram + statystyki) dla kluczowej miary.”",
            "trigger": "Uruchamia rodzinę odpowiedzi skupioną na rozrzucie, medianie, percentylach, outlierach i kształcie rozkładu.",
            "output": "Histogram, boxplot, violin, percentyle, ECDF i wykresy outlierów.",
        },
        {
            "branch": "Composition Static",
            "question": "„Pokaż sumaryczną wartość sprzedaży i per kategoria.”",
            "trigger": "Uruchamia gałąź do pokazania udziałów, liderów i wkładu kategorii w wynik bez osi czasu.",
            "output": "Struktura, udziały, top-N, liderzy kategorii i ranking udziałów.",
        },
        {
            "branch": "Composition Over Time",
            "question": "„Pokaż sprzedaż per kategorię w czasie (cały okres).”",
            "trigger": "Uruchamia odpowiedź pokazującą, jak zmieniają się wartości i udziały kategorii na osi czasu.",
            "output": "Linie, stacked shares, area charts i trend całego okresu.",
        },
        {
            "branch": "Comparison",
            "question": "„Porównaj kategorie: kto jest liderem, kto odstaje?”",
            "trigger": "Uruchamia gałąź do czytelnego porównania pozycji, różnic i odstępstw między elementami.",
            "output": "Rankingi, słupki, lider / outsider i porównanie pozycji.",
        },
        {
            "branch": "Relationship",
            "question": "„Czy istnieje zależność między dwiema zmiennymi (korelacja / trend)?”",
            "trigger": "Uruchamia odpowiedź skoncentrowaną na kierunku, sile i kształcie relacji między zmiennymi.",
            "output": "Scatter plot, trend, korelacja i zależność dwóch zmiennych.",
        },
        {
            "branch": "Quality / Sanity",
            "question": "„Jakie są braki danych i które kolumny są najbardziej problematyczne?”",
            "trigger": "Przełącza Data Chat na logikę sanity-checku: braki, duplikaty, anomalie i kolumny wymagające uwagi.",
            "output": "Braki, heatmapy, duplikaty, alerty i kolumny ryzyka.",
        },
        {
            "branch": "Segmentation / Clusters",
            "question": "„Pokaż wielkość segmentów/klastrów (liczebność) oraz ich udział.”<br>„Opisz cechy wyróżniające segmenty/klastry i zaproponuj ich nazwy.”",
            "trigger": "Uruchamia odpowiedź zorientowaną na liczebność segmentów, profile klastrów oraz ich interpretację biznesową.",
            "output": "Liczebność segmentów, profile, cechy wyróżniające i nazwy segmentów.",
        },
    ]

    cards = []
    for row in rows:
        cards.append(
            (
                '<div class="dc-router-card">'
                '<div class="dc-router-head"><span class="dc-router-dot"></span>{branch}</div>'
                '<div class="dc-router-grid">'
                '<div class="dc-router-block">'
                '<span class="dc-router-label">Przykładowe pytanie użytkownika</span>'
                '<div class="dc-router-question">{question}</div>'
                '</div>'
                '<div class="dc-router-block">'
                '<span class="dc-router-label">Co to pytanie uruchamia</span>'
                '<div class="dc-router-trigger">{trigger}</div>'
                '</div>'
                '<div class="dc-router-block">'
                '<span class="dc-router-label">Co zwykle dostajesz</span>'
                '<div class="dc-router-output">{output}</div>'
                '</div>'
                '</div>'
                '</div>'
            ).format(**row)
        )

    st.markdown(
        '<div class="dc-router-list">' + ''.join(cards) + '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="dc-router-note"><strong>W praktyce:</strong> jeśli zmienisz pytanie z „per kategoria” na „per kategoria w czasie”, '
        'to Data Chat przełączy się na inną rodzinę odpowiedzi i pokaże inny zestaw wykresów.</div>',
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="AUTO EDA FOR",
    page_icon="\U0001F9E0",
    layout="wide",
    initial_sidebar_state="expanded",
)

hide_default_multipage_nav()
render_flow_nav(current_id=None)

st.title("AUTO EDA FOR")
st.markdown(
    "AUTO EDA FOR prowadzi u\u017cytkownika przez pe\u0142ny potok pracy z danymi: "
    "od wgrania danych i sanity checku, przez automatyczn\u0105 diagnoz\u0119 i Data Chat, "
    "a\u017c do trenowania modelu i predykcji."
)

st.info(
    "\U0001F449 Poruszaj si\u0119 wy\u0142\u0105cznie po potoku etap\u00f3w u g\u00f3ry. "
    "Ka\u017cdy etap ma w\u0142asne ustawienia wewn\u0105trz swojej strony."
)

_render_home_journey_section()

st.markdown("---")

st.subheader("\U0001F9E0 Co aplikacja robi za Ciebie")
st.markdown(
    """
- automatycznie wykrywa typ zadania i rol\u0119 kolumn,
- wykrywa ryzykowne kolumny, braki, duplikaty i outliery,
- potrafi prze\u0142\u0105czy\u0107 si\u0119 mi\u0119dzy zadaniami, je\u015bli dataset jest wielozadaniowy,
- zapisuje artefakty danych, \u017ceby kolejne etapy dzia\u0142a\u0142y bez ponownego uploadu.
    """
)

st.subheader("\U0001F9ED Jak przej\u015b\u0107 przez aplikacj\u0119")
st.markdown(
    """
1. **Analiza Danych**: wgraj plik albo wybierz dane demo i wykonaj pe\u0142ne przeliczenie.
2. **Automat EDA**: odbierz szybki audyt jako\u015bci, cleaning i rekomendacje.
3. **Data Chat**: zadawaj pytania do danych i ogl\u0105daj interpretacje oraz wykresy.
4. **Trenowanie modelu**: uruchom Auto-ML na przygotowanych danych.
5. **Predykcja**: wykonaj predykcj\u0119 batchow\u0105 lub punktow\u0105 i pobierz wyniki.
    """
)

st.success(
    "To wszystko. Kliknij Etap 1 w potoku u g\u00f3ry i zacznij od wgrania danych \U0001F642"
)

_render_datachat_question_router_section()
