/* ---------------- sliding the orders: dense pattern sliders and maps (D.patterns, simulateBoundaryHarmonicRingCodePatterns11x11.py) ---------------- */
(function patternSlider(){
  if(!D.patterns) return;
  ['psTitle','psIntro','patternSlider'].forEach(id=>{document.getElementById(id).style.display='';});
  /* where each order alone can move before the stripe is lost, from the dense sliders */
  (function windows(){
    const Q=D.patterns,F0=Q.families,tr=Q.trainedCoefficients,parts=[];
    for(let o=0;o<tr.length;o++){
      const stops=[];F0.zoomSlider.order.forEach((q,i)=>{if(q===o)stops.push(i);});
      const formed=stops.filter(i=>F0.zoomSlider.overlapAtRead[i]>=0.9).map(i=>F0.zoomSlider.coefficients[i][o]);
      const wide=[];F0.wideSlider.order.forEach((q,i)=>{if(q===o)wide.push(i);});
      const wideFormed=wide.filter(i=>F0.wideSlider.overlapAtRead[i]>=0.9).length;
      parts.push(formed.length?`order ${o}: the stripe is formed from ${Math.min(...formed).toFixed(3)} to ${Math.max(...formed).toFixed(3)} (${formed.length} of the ${stops.length} stops within ±0.06 of the trained ${tr[o].toFixed(3)}, ${wideFormed} of ${wide.length} over the whole range)`:`order ${o}: no stop within ±0.06 forms the stripe`);
    }
    const el0=document.getElementById('psWindowText');el0.style.display='';
    el0.innerHTML=`<strong>How far each order can move on its own</strong> and still leave the stripe at iteration ${Q.readIteration} (overlap 0.9 or more): ${parts.join('; ')}.`;
  })();
  const SETS={stripe:D.patterns,face:D.patternsFace};
  let P=SETS.stripe,F,trained,READ,basis,FEATURE,NAME,NTARGET;
  function useSet(key){P=SETS[key];F=P.families;trained=P.trainedCoefficients;READ=P.readIteration;basis=P.ringBasis;FEATURE=new Set(P.featureCells||D.targetCells);NAME=key==='face'?'face':'stripe';NTARGET=FEATURE.size;}
  useSet('stripe');
  const $=id=>document.getElementById(id), clear=n=>{while(n.firstChild)n.removeChild(n.firstChild);};
  const darkInterior=v=>{const out=[];v.forEach((m,i)=>{const r=Math.floor(i/COLUMNS),c=i%COLUMNS;if(r>0&&r<ROWS-1&&c>0&&c<COLUMNS-1&&m<thresholdMilliVolts)out.push(i);});return out;};
  const changed=(a,b)=>{const A=new Set(darkInterior(a)),B=new Set(darkInterior(b));let n=0;A.forEach(i=>{if(!B.has(i))n++;});B.forEach(i=>{if(!A.has(i))n++;});return n;};
  const ringValuesOf=co=>basis.map(row=>Math.min(2,Math.max(0,row.reduce((s,b,o)=>s+b*co[o],0))));
  const sgn=v=>(v>=0?'+':'−')+Math.abs(v).toFixed(3);
  const showTissue=(svgId,values,captionId,caption)=>{const g=$(svgId);clear(g);
    if(NAME==='stripe')thumb(g,values,6,6,19,1.2);
    else{for(let r=0;r<ROWS;r++)for(let c=0;c<COLUMNS;c++)el('rect',{x:6+c*19,y:6+r*19,width:17.8,height:17.8,fill:'var(--seq)','fill-opacity':vmemOpacity(values[r*COLUMNS+c])},g);
      FEATURE.forEach(i=>el('rect',{x:6+(i%COLUMNS)*19-0.3,y:6+Math.floor(i/COLUMNS)*19-0.3,width:18.4,height:18.4,fill:'none',stroke:'var(--ochre)','stroke-width':0.9,'stroke-opacity':0.9,'pointer-events':'none'},g));}
    if(captionId)$(captionId).textContent=caption;};
  const showRing=(svgId,co,captionId)=>{const g=$(svgId);clear(g);const v=ringValuesOf(co);ringCode(g,v,6,6,19,2);if(captionId)$(captionId).textContent=`ring held: ${Math.min(...v).toFixed(2)} to ${Math.max(...v).toFixed(2)}`+(Math.max(...v)>bistableUpperEdge?' · above 1.439 in '+v.filter(x=>x>bistableUpperEdge).length+' cells':'');};

  /* ------------------------------------------------ the slider */
  const setSel=$('psSet'),orderSel=$('psOrder'),rangeSel=$('psRange'),slider=$('psSlider'),readout=$('psReadout'),chart=$('psChart'),stats=$('psStats');
  if(!SETS.face)document.getElementById('psSetBox').style.display='none';
  function fillOrders(){clear(orderSel);const names=['the dial','top against bottom','the oval','the third harmonic'];trained.forEach((_,o)=>{const op=document.createElement('option');op.value=String(o);op.textContent=`${o} (a${o}, ${names[o]||'harmonic'})`;orderSel.appendChild(op);});}
  fillOrders();
  let stops=[],xs=[],cur=0;
  function build(){
    const order=+orderSel.value,fam=F[rangeSel.value];
    stops=[];fam.order.forEach((o,i)=>{if(o===order)stops.push(i);});
    xs=stops.map(i=>fam.coefficients[i][order]);
    slider.max=String(stops.length-1);
    let k=0,best=1e9;xs.forEach((x,j)=>{const d=Math.abs(x-trained[order]);if(d<best){best=d;k=j;}});
    cur=k;slider.value=String(cur);
    drawChart();update();
  }
  function update(){
    const order=+orderSel.value,fam=F[rangeSel.value],i=stops[cur],co=fam.coefficients[i];
    showTissue('psReadPanel',fam.vmemAtRead[i],'psReadCaption',`iteration ${READ} · overlap ${fam.overlapAtRead[i].toFixed(2)} · ${fam.strayAtRead[i]} strays`);
    showTissue('psBestPanel',fam.vmemAtBest[i],'psBestCaption',`best moment, iteration ${fam.bestIteration[i]} · overlap ${fam.overlapAtBest[i].toFixed(2)} · ${fam.strayAtBest[i]} strays`);
    showRing('psRingPanel',co,'psRingCaption');
    const off=co[order]-trained[order];
    readout.textContent=`a${order} = ${co[order].toFixed(4)} · ${Math.abs(off)<5e-4?'the trained value':sgn(off)+' from trained'} · score ${fam.score[i].toFixed(2)} mV`+(fam.clippedShare[i]>0?` · ${Math.round(fam.clippedShare[i]*40)} of 40 ring cells clipped`:'');
    marker();
  }
  let markerLine=null,chartGeom=null;
  function drawChart(){
    clear(chart);
    const order=+orderSel.value,fam=F[rangeSel.value],W=900,left=64,right=24,x0=left,x1=W-right;
    const xmin=Math.min(...xs),xmax=Math.max(...xs),X=v=>x0+(v-xmin)/(xmax-xmin||1)*(x1-x0);
    const stripeDark=stops.map(i=>darkInterior(fam.vmemAtRead[i]).filter(c=>FEATURE.has(c)).length);
    const panels=[{y0:14,y1:130,name:'overlap with the '+NAME,lo:0,hi:1,ticks:[0,0.5,0.9,1]},
                  {y0:162,y1:240,name:'dark cells at '+READ,lo:0,hi:Math.max(27,...stops.map(i=>fam.strayAtRead[i]),...stripeDark),ticks:null},
                  {y0:272,y1:320,name:'score, mV',lo:0,hi:Math.max(25,...stops.map(i=>fam.score[i])),ticks:null}];
    chartGeom={X,xmin,xmax,x0,x1};
    panels.forEach(p=>{p.Y=v=>p.y1-(v-p.lo)/(p.hi-p.lo)*(p.y1-p.y0);
      el('rect',{x:x0,y:p.y0,width:x1-x0,height:p.y1-p.y0,fill:'none',stroke:'var(--line)','stroke-width':0.8},chart);
      const cy=(p.y0+p.y1)/2;tx(p.name,{x:14,y:cy,'text-anchor':'middle','font-size':10.5,transform:`rotate(-90 14 ${cy})`},chart);
      const tk=p.ticks||[0,Math.round(p.hi/2),Math.round(p.hi)];
      tk.forEach(t=>{tx(String(t),{x:x0-6,y:p.Y(t)+3.5,'text-anchor':'end','font-size':9.5,'font-family':MONO},chart);el('line',{x1:x0,x2:x1,y1:p.Y(t),y2:p.Y(t),stroke:'var(--line-soft)','stroke-width':0.6},chart);});});
    /* the formed band: overlap at the readout at or above 0.9, each stop owning the half-way points to its neighbours */
    xs.forEach((x,j)=>{if(fam.overlapAtRead[stops[j]]<0.9)return;
      const a=j>0?(xs[j-1]+x)/2:x,b=j<xs.length-1?(x+xs[j+1])/2:x;
      el('rect',{x:X(a),y:panels[0].y0,width:Math.max(1.5,X(b)-X(a)),height:panels[0].y1-panels[0].y0,fill:'var(--ochre)','fill-opacity':0.25},chart);});
    const line=(vals,p,attrs)=>el('path',Object.assign({d:vals.map((v,j)=>`${j?'L':'M'}${X(xs[j]).toFixed(1)} ${p.Y(v).toFixed(1)}`).join(' '),fill:'none','stroke-linejoin':'round'},attrs),chart);
    const pick=key=>stops.map(i=>fam[key][i]);
    line(pick('overlapAtBest'),panels[0],{stroke:'var(--ink-3)','stroke-width':1.3,'stroke-dasharray':'4 3'});
    line(pick('overlapAtRead'),panels[0],{stroke:'var(--teal)','stroke-width':2});
    line(pick('strayAtRead'),panels[1],{stroke:'var(--rose)','stroke-width':1.6});
    line(stripeDark,panels[1],{stroke:'var(--teal)','stroke-width':1.8});
    line(pick('score'),panels[2],{stroke:'var(--ochre)','stroke-width':1.8});
    tx(`${NAME} cells dark (of ${NTARGET})`,{x:x1-4,y:panels[1].y0+13,'text-anchor':'end','font-size':10,fill:'var(--teal)'},chart);tx('strays',{x:x1-4,y:panels[1].y0+26,'text-anchor':'end','font-size':10,fill:'var(--rose)'},chart);
    tx('overlap at '+READ,{x:x1-4,y:panels[0].y0+13,'text-anchor':'end','font-size':10,fill:'var(--teal)'},chart);tx('at the best moment (dashed)',{x:x1-4,y:panels[0].y0+26,'text-anchor':'end','font-size':10},chart);
    /* x axis */
    const nt=6;for(let t=0;t<=nt;t++){const v=xmin+(xmax-xmin)*t/nt;el('line',{x1:X(v),x2:X(v),y1:panels[2].y1,y2:panels[2].y1+4,stroke:'var(--ink-3)'},chart);tx(v.toFixed(rangeSel.value==='zoomSlider'?3:2),{x:X(v),y:panels[2].y1+16,'text-anchor':'middle','font-size':9.5,'font-family':MONO},chart);}
    tx('coefficient of order '+order,{x:x1,y:panels[2].y1+31,'text-anchor':'end','font-size':10.5},chart);
    const tv=trained[order];if(tv>=xmin&&tv<=xmax)el('line',{x1:X(tv),x2:X(tv),y1:panels[0].y0,y2:panels[2].y1,stroke:'var(--ink-2)','stroke-width':1,'stroke-dasharray':'2 3'},chart);
    markerLine=el('line',{x1:0,x2:0,y1:panels[0].y0,y2:panels[2].y1,stroke:'var(--ink)','stroke-width':1.4},chart);
    const hit=el('rect',{x:x0,y:panels[0].y0,width:x1-x0,height:panels[2].y1-panels[0].y0,fill:'transparent',style:'cursor:pointer'},chart);
    const nearest=ev=>{const r=chart.getBoundingClientRect(),x=(ev.clientX-r.left)/r.width*W,v=xmin+(x-x0)/(x1-x0)*(xmax-xmin);let k=0,b=1e9;xs.forEach((q,j)=>{const d=Math.abs(q-v);if(d<b){b=d;k=j;}});return k;};
    hit.addEventListener('click',ev=>{cur=nearest(ev);slider.value=String(cur);update();});
    hook(hit,()=>'');
    hit.addEventListener('mousemove',ev=>{const j=nearest(ev),i=stops[j];tip.innerHTML=`a${order} = ${xs[j].toFixed(4)}<br>overlap at ${READ}: ${fam.overlapAtRead[i].toFixed(2)}<br>strays at ${READ}: ${fam.strayAtRead[i]}<br>score ${fam.score[i].toFixed(2)} mV, best moment ${fam.bestIteration[i]}`;});
    summary();
  }
  function marker(){if(!markerLine||!chartGeom)return;const x=chartGeom.X(xs[cur]);markerLine.setAttribute('x1',x);markerLine.setAttribute('x2',x);}
  function summary(){
    const fam=F[rangeSel.value],order=+orderSel.value;let jumps=0,maxJump=0,big=0;
    for(let j=1;j<stops.length;j++){const d=changed(fam.vmemAtRead[stops[j-1]],fam.vmemAtRead[stops[j]]);if(d>0)jumps++;if(d>=4)big++;maxJump=Math.max(maxJump,d);}
    const formed=stops.filter(i=>fam.overlapAtRead[i]>=0.9);
    const fx=formed.map(i=>fam.coefficients[i][order]);
    const darkCounts=stops.map(i=>darkInterior(fam.vmemAtRead[i]).length);
    stats.innerHTML=`Along this slider (${stops.length} stops): the set of dark interior cells at iteration ${READ} changes in <b>${jumps}</b> of ${stops.length-1} steps, by at most <b>${maxJump}</b> cells in one step, and by 4 cells or more in <b>${big}</b> of them; `
      +`the number of dark cells ranges from ${Math.min(...darkCounts)} to ${Math.max(...darkCounts)}. `
      +(formed.length?`The ${NAME} (overlap 0.9 or more) is formed at <b>${formed.length}</b> ${formed.length===1?'stop, at':'stops, between'} ${Math.min(...fx).toFixed(4)}${formed.length===1?'':' and '+Math.max(...fx).toFixed(4)}.`:`The ${NAME} is formed at none of these stops.`);
  }
  function captions(){$('psCaptionText').innerHTML=`Top: overlap with the ${NAME} (1 is the ${NAME} and nothing else) at iteration ${READ} (solid) and at each code’s own best moment (dashed); the ochre band is overlap 0.9 or more. Middle: dark interior cells at ${READ}, those inside the ${NAME} and those outside it (strays). Bottom: the balanced RMS at each code’s best moment (the training score, lower is better). The dotted vertical line is the trained value; click the chart to move the slider. Tissue panels: darker is more hyperpolarised, the ochre frame is the ${NAME}; the ring panel is the code held during the hold, darker is a higher G_pol / G_ref (0 to 2).`;}
  setSel.addEventListener('change',()=>{useSet(setSel.value);fillOrders();$('psMapFigure').style.display=NAME==='stripe'?'':'none';rangeSel.value='wideSlider';captions();build();});
  captions();
  orderSel.addEventListener('change',build);rangeSel.addEventListener('change',build);
  slider.addEventListener('input',()=>{cur=+slider.value;update();});
  build();

  /* ------------------------------------------------ the maps */
  const famSel=$('psMapFamily'),colSel=$('psMapColour'),map=$('psMap'),mapStats=$('psMapStats');
  const AVAILABLE=['mapA0A2','mapTopSide'].filter(k=>F[k]);
  [...famSel.options].forEach(o=>{if(!F[o.value])o.remove();});
  let selected=0;
  const mapCaptions={mapA0A2:'The ring code a0 + a2 cos 2θ over the whole plane (a1 = 0). The stripe forms along a thin line, not over a region.',mapTopSide:'The same plane as the ring’s level at the top and bottom, T = a0 + a2, against its level at the sides, S = a0 − a2, zoomed on the window; T runs across in steps of 0.002.'};
  function drawMap(){
    const key=famSel.value,fam=F[key],[ax,ay]=fam.axes,nx=ax.length,ny=ay.length,colour=colSel.value;
    clear(map);
    const W=560,H=480,left=54,top=14,pw=W-left-20,ph=H-top-50,cw=pw/nx,ch=ph/ny;
    const vals=fam[colour],vmax=colour==='overlapAtRead'||colour==='overlapAtBest'?1:Math.max(...vals,1),vmin=colour==='bestIteration'?Math.min(...vals):0;
    for(let i=0;i<nx;i++)for(let j=0;j<ny;j++){
      const idx=i*ny+j,v=vals[idx],formed=fam.overlapAtRead[idx]>=0.9;
      const rect=el('rect',{x:left+i*cw,y:top+(ny-1-j)*ch,width:Math.max(cw-0.4,0.6),height:Math.max(ch-0.4,0.6),fill:(colour.startsWith('overlap')&&v>=0.9)?'var(--ochre)':colour==='strayAtRead'?'var(--rose)':'var(--teal)','fill-opacity':0.06+0.9*(v-vmin)/((vmax-vmin)||1)},map);
      hook(rect,()=>`${key==='mapTopSide'?'T':'a0'} ${ax[i]} · ${key==='mapTopSide'?'S':'a2'} ${ay[j]}<br>overlap at ${READ}: ${fam.overlapAtRead[idx].toFixed(2)}, at best ${fam.overlapAtBest[idx].toFixed(2)}<br>strays at ${READ}: ${fam.strayAtRead[idx]} · best moment ${fam.bestIteration[idx]}`);
      rect.addEventListener('click',()=>{selected=idx;showCell();drawMap();});
    }
    /* axes */
    const labelX=key==='mapTopSide'?'T = a0 + a2 (ring level at top and bottom)':'a0',labelY=key==='mapTopSide'?'S = a0 − a2 (sides)':'a2';
    for(let t=0;t<=5;t++){const i=Math.round(t*(nx-1)/5);tx(ax[i].toFixed(key==='mapTopSide'?3:2),{x:left+(i+0.5)*cw,y:top+ph+14,'text-anchor':'middle','font-size':9.5,'font-family':MONO},map);}
    for(let t=0;t<=4;t++){const j=Math.round(t*(ny-1)/4);tx(ay[j].toFixed(2),{x:left-6,y:top+(ny-1-j+0.5)*ch+3.5,'text-anchor':'end','font-size':9.5,'font-family':MONO},map);}
    tx(labelX,{x:left+pw/2,y:H-8,'text-anchor':'middle','font-size':11},map);tx(labelY,{x:12,y:top+ph/2,'text-anchor':'middle','font-size':11,transform:`rotate(-90 12 ${top+ph/2})`},map);
    /* the trained code, and the selected cell */
    const t0=key==='mapTopSide'?trained[0]+trained[2]:trained[0],t1=key==='mapTopSide'?trained[0]-trained[2]:trained[2];
    const nearestIdx=(arr,v)=>{let k=0,b=1e9;arr.forEach((a,i)=>{const d=Math.abs(a-v);if(d<b){b=d;k=i;}});return k;};
    const ti=nearestIdx(ax,t0),tj=nearestIdx(ay,t1);
    el('rect',{x:left+ti*cw-1,y:top+(ny-1-tj)*ch-1,width:cw+2,height:ch+2,fill:'none',stroke:'var(--ink)','stroke-width':1.6,'stroke-dasharray':'3 2'},map);
    const si=Math.floor(selected/ny),sj=selected%ny;
    el('rect',{x:left+si*cw-1,y:top+(ny-1-sj)*ch-1,width:cw+2,height:ch+2,fill:'none',stroke:'var(--rose)','stroke-width':2},map);
    tx(`dashed black: the trained code · red: the cell on the right`,{x:left+pw,y:top+ph+30,'text-anchor':'end','font-size':9.5},map);
    $('psMapCaptionText').textContent=mapCaptions[key]+' Shade is the quantity chosen; ochre is overlap 0.9 or more. Click a cell to see its tissue.';
  }
  function showCell(){
    const fam=F[famSel.value],idx=selected,co=fam.coefficients[idx];
    showTissue('psMapReadPanel',fam.vmemAtRead[idx],'psMapReadCaption',`iteration ${READ} · overlap ${fam.overlapAtRead[idx].toFixed(2)}`);
    showTissue('psMapBestPanel',fam.vmemAtBest[idx],'psMapBestCaption',`best moment ${fam.bestIteration[idx]} · overlap ${fam.overlapAtBest[idx].toFixed(2)}`);
    mapStats.innerHTML=`a0 ${co[0].toFixed(4)}, a1 ${co[1].toFixed(4)}, a2 ${co[2].toFixed(4)} · ring ${fam.ringMin[idx].toFixed(2)} to ${fam.ringMax[idx].toFixed(2)} · score ${fam.score[idx].toFixed(2)} mV · ${fam.strayAtRead[idx]} strays at ${READ}.`;
  }
  function chooseTrained(){
    const key=famSel.value,fam=F[key],[ax,ay]=fam.axes;
    const t0=key==='mapTopSide'?trained[0]+trained[2]:trained[0],t1=key==='mapTopSide'?trained[0]-trained[2]:trained[2];
    const near=(arr,v)=>{let k=0,b=1e9;arr.forEach((a,i)=>{const d=Math.abs(a-v);if(d<b){b=d;k=i;}});return k;};
    selected=near(ax,t0)*ay.length+near(ay,t1);
  }
  if(AVAILABLE.length){famSel.addEventListener('change',()=>{chooseTrained();drawMap();showCell();});colSel.addEventListener('change',drawMap);chooseTrained();drawMap();showCell();}
  else $('psMapFigure').style.display='none';
})();

/* ---------------- the smoothness table: stripe against face, from analyzeBoundaryHarmonicRingCodePatternSmoothness11x11.py ---------------- */
(function smoothness(){
  const S=D.smoothness; if(!S) return;
  document.getElementById('psSmoothFigure').style.display='';
  const rangeSel=document.getElementById('psSmoothRange'),momentSel=document.getElementById('psSmoothMoment'),table=document.getElementById('psSmoothTable'),text=document.getElementById('psSmoothText');
  const pct=x=>Math.round(100*x)+'%',mean=a=>a.reduce((s,x)=>s+x,0)/a.length;
  function render(){
    const fam=rangeSel.value,view=momentSel.value,rows=[],agg={};
    ['stripe','face'].forEach(set=>{
      const entries=Object.entries(S[set].families[fam]).map(([o,e])=>[o,e[view]]);
      agg[set]={changed:mean(entries.map(([,m])=>m.changedShare)),mean:mean(entries.map(([,m])=>m.meanChange)),max:Math.max(...entries.map(([,m])=>m.maxChange)),big:mean(entries.map(([,m])=>m.bigJumpShare)),sets:mean(entries.map(([,m])=>m.distinctSets)),ov:mean(entries.map(([,m])=>m.meanOverlapStep)),ovMax:Math.max(...entries.map(([,m])=>m.maxOverlapStep)),formed:mean(entries.map(([,m])=>m.formedShare))};
      entries.forEach(([o,m])=>rows.push(`<tr><td>${set}</td><td class="mono">${o}</td><td class="mono">${m.step.toFixed(4)}</td><td class="mono">${pct(m.changedShare)}</td><td class="mono">${m.meanChange.toFixed(2)}</td><td class="mono">${m.maxChange}</td><td class="mono">${pct(m.bigJumpShare)}</td><td class="mono">${m.distinctSets}</td><td class="mono">${m.meanOverlapStep.toFixed(3)}</td><td class="mono">${m.maxOverlapStep.toFixed(2)}</td><td class="mono">${pct(m.formedShare)}</td></tr>`));
    });
    table.innerHTML='<tr><th>code</th><th>order</th><th>step</th><th>steps with any change</th><th>cells changed per step, mean</th><th>most cells in one step</th><th>steps changing 4+ cells</th><th>distinct dark sets</th><th>|Δ overlap| per step, mean</th><th>largest</th><th>stops with the pattern formed</th></tr>'+rows.join('');
    const a=agg.stripe,b=agg.face,moment=view==='atRead'?'at the readout':'at each code’s own best moment';
    text.innerHTML=`${fam==='zoomSlider'?'Per step of 0.001':'Over the whole range'}, ${moment}: the stripe code’s dark set changes in <b>${pct(a.changed)}</b> of steps, by <b>${a.mean.toFixed(2)}</b> cells on average and at most <b>${a.max}</b> in one step (${pct(a.big)} of steps change 4 or more); the face code’s changes in <b>${pct(b.changed)}</b> of steps, by <b>${b.mean.toFixed(2)}</b> cells on average and at most <b>${b.max}</b> (${pct(b.big)} change 4 or more). `
      +`The overlap with the target moves by <b>${a.ov.toFixed(3)}</b> per step for the stripe and <b>${b.ov.toFixed(3)}</b> for the face, at most <b>${a.ovMax.toFixed(2)}</b> and <b>${b.ovMax.toFixed(2)}</b>. A slider passes through about <b>${b.sets.toFixed(0)}</b> different dark sets for the face code and <b>${a.sets.toFixed(0)}</b> for the stripe code.`;
  }
  rangeSel.addEventListener('change',render);momentSel.addEventListener('change',render);render();
})();
