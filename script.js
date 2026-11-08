(async function(){
    const $=id=>document.getElementById(id),mix=(a,b,t)=>a+(b-a)*t;
    const L=[5,10,20,30,40,50],S={personal:[.610,.702,.806,.859,.878,.886],population:[.582,.672,.847,.934,.966,.975],hybrid:[.619,.733,.889,.952,.978,.987]};
    const reduce=matchMedia('(prefers-reduced-motion: reduce)').matches;
    function auroc(a,p){const q=Math.min(50,Math.max(5,p));let i=0;while(i<4&&q>L[i+1])i++;return mix(a[i],a[i+1],(q-L[i])/(L[i+1]-L[i]))}
    let gain=20,who='all',walker=null,D=null,burst=0,paintCols=()=>{},moveMarker=()=>{};
    
    // ---------- data ----------
    try{const r=await fetch('pca_data.json');if(!r.ok)throw 0;D=await r.json();$('status').textContent=`Real data: ${D.base.length.toLocaleString()} walking windows, PCA of 24 features`}
    catch(e){$('status').textContent='No pca_data.json found. Run python export_pca.py, then serve this folder with python -m http.server.'}
    
    // ---------- AUROC chart ----------
    const svg=$('chart'),NS='http://www.w3.org/2000/svg',cl=55,cr=805,ct=24,cb=278;
    const X=v=>cl+(v-5)/45*(cr-cl),Y=v=>cb-(v-.5)/.5*(cb-ct);
    function el(t,a,txt){const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);if(txt!==undefined)e.textContent=txt;svg.appendChild(e);return e}
    [.5,.6,.7,.8,.9,1].forEach(v=>{el('line',{x1:cl,y1:Y(v),x2:cr,y2:Y(v),stroke:'rgba(170,190,240,.16)','stroke-dasharray':'4 7'});el('text',{x:14,y:Y(v)+4,fill:'#9ba9c6','font-size':12},v.toFixed(1))});
    L.forEach(v=>el('text',{x:X(v),y:308,'text-anchor':'middle',fill:'#9ba9c6','font-size':12},v+'%'));
    const COL={personal:'#78e6ff',population:'#b6a0ff',hybrid:'#70f3c1'};
    for(const k in S){el('polyline',{points:S[k].map((v,i)=>X(L[i])+','+Y(v)).join(' '),fill:'none',stroke:COL[k],'stroke-width':k==='hybrid'?4:3,'stroke-linejoin':'round'});
    S[k].forEach((v,i)=>{const c=el('circle',{cx:X(L[i]),cy:Y(v),r:5,fill:COL[k],stroke:'#0c1328','stroke-width':2});const t=document.createElementNS(NS,'title');t.textContent=`${k}: ${v.toFixed(3)} at ${L[i]}%`;c.appendChild(t)})}
    const mk=el('line',{y1:ct,y2:cb,stroke:'#ff7b87','stroke-width':2,'stroke-dasharray':'5 5'});
    
    // ---------- gain controls ----------
    let demo=0;function stopDemo(){if(demo){cancelAnimationFrame(demo);demo=0;$('play').textContent='▶ Play the demo'}}
    const CAP=g=>g<15?'A tiny change. The red dots barely leave the crowd, so every AI is close to guessing.':g<30?'A noticeable change. The red dots drift away and the blended (hybrid) AI pulls ahead.':'A big change. The red dots are far from normal, so the AIs spot it easily and the hybrid scores near-perfect.';
    function setGain(v){const nv=Math.round(+v);burst=Math.min(1.2,burst+Math.abs(nv-gain)*.15);gain=nv;$('gain1').value=$('gain2').value=gain;$('pct').textContent=$('pct2').textContent=gain+'%';
     const r={};[['P','personal'],['G','population'],['H','hybrid']].forEach(([id,k])=>{const x=auroc(S[k],gain);r[id]=x;$('v'+id).textContent=x.toFixed(3);$('b'+id).style.width=x*100+'%'});
     mk.setAttribute('x1',X(gain));mk.setAttribute('x2',X(gain));
     $('takeaway').textContent=`At ${gain}% stronger movement: blended ${r.H.toFixed(3)}, everyone-else ${r.G.toFixed(3)}, one-person ${r.P.toFixed(3)}. Higher is better.`;$('caption').textContent=CAP(gain)}
    ['gain1','gain2'].forEach(id=>$(id).addEventListener('input',e=>{stopDemo();setGain(e.target.value)}));
    let drag=false;const fromChart=e=>{const bx=svg.getBoundingClientRect(),x=(e.clientX-bx.left)/bx.width*850;setGain(Math.min(50,Math.max(5,5+(x-cl)/(cr-cl)*45)))};
    svg.style.cursor='ew-resize';svg.style.touchAction='none';
    svg.addEventListener('pointerdown',e=>{stopDemo();drag=true;svg.setPointerCapture(e.pointerId);fromChart(e)});
    svg.addEventListener('pointermove',e=>{if(drag)fromChart(e)});
    svg.addEventListener('pointerup',()=>drag=false);
    $('play').onclick=()=>{if(demo)return stopDemo();$('lab').scrollIntoView({behavior:'smooth'});$('play').textContent='❚❚ Pause';const t0=performance.now(),step=t=>{const p=Math.min(1,(t-t0)/9000);setGain(5+45*p);if(p<1)demo=requestAnimationFrame(step);else{demo=0;$('play').textContent='▶ Play again'}};demo=requestAnimationFrame(step)};
    setGain(20);
    
    // ---------- experiment atlas ----------
    const EX=[
    ['Personalized baseline','sur','1 → 21','one person, then everyone','Can an AI learn one person’s walk and notice when it changes?','Yes, for the first person I tried. But when I tested all 21 people, that success didn’t carry over.'],
    ['Multi-person test','sur','4 of 21','people where learning your own walk beat learning the crowd','Is it better to learn each person separately than to learn from everyone?','Usually not. For bigger changes, the AI trained on everyone else noticed more, for most people.'],
    ['Types of change','cau','4 kinds','of fake change tested','Does it matter how the walk changes?','Yes. How well the AI noticed depended on what kind of change I made and how big it was.'],
    ['Same amount of data','sur','Equal data','the crowd AI still came out ahead','Is the crowd AI only better because it saw more data?','No. Even when both AIs saw the same amount of data, the crowd AI kept its edge.'],
    ['Scoring without a cutoff','win','0.975','crowd AI score at 50% change','Can I score the AI fairly without picking one alarm cutoff?','Yes, using a score where 1.0 is perfect and 0.5 is guessing. The crowd AI separated big changes best.'],
    ['Person-by-person','sur','0.979 vs 0.375','Participant 27: own-walk AI vs crowd AI','Is the best AI the same for everyone?','Not at all. For Participant 27, learning their own walk worked far better. For Participant 17 it was the reverse (0.976 crowd vs 0.592 own).'],
    ['Blending both AIs','win','6 of 6','change levels where the hybrid won','What if I combine the two AIs?','The blend had the best average score at every change level I tested.'],
    ['Could it be luck?','cau','5–10%','small changes: strongest evidence','Could the hybrid’s win just be luck?','For small changes the win looks reliable. For big changes it’s less certain, so I don’t over-claim.'],
    ['Which sensors matter','sur','0.897 vs 0.838','gyroscope alone vs both sensors','Does adding more sensors always help?','No. For rotation changes, the gyroscope alone beat both sensors together. When movement and rotation both changed, using both won (0.935).'],
    ['When a sensor breaks','cau','0.602','score for slow gyroscope drift','What if the sensor itself starts to fail?','Sudden noise was easy to catch, but slow drift looked like a normal walk. A real device would need a separate sensor-health check.'],
    ['False alarms','cau','7.6%','false alarms vs a 5% goal','Can I control how often the AI cries wolf?','Not reliably. I aimed for 5% false alarms and got 7.6% on new data. At a 20% change it caught 46.2% of changed walks.'],
    ['Walking fingerprint','sur','92.2%','correctly guessed who was walking','Is a person’s walk like a fingerprint?','It looked that way: 92.2% accuracy at first. The next experiment tests whether that’s real.'],
    ['Overlap check','cau','51.1%','accuracy after removing near-copies','Was 92.2% too good to be true?','Partly. Neighboring clips overlap, so the AI was matching near-copies. Removing them dropped accuracy to 51.1%, still well above chance but much lower.']];
    const TAG={win:['Confirmed','var(--mint)'],sur:['Surprising','var(--purple)'],cau:['Be careful','var(--coral)']};
    const list=$('exList');
    EX.forEach((e,i)=>{const b=document.createElement('button');b.setAttribute('role','tab');b.innerHTML=`<i style="background:${TAG[e[1]][1]}"></i>${e[0]}`;b.onclick=()=>pick(i);list.appendChild(b)});
    function pick(i){const e=EX[i],t=TAG[e[1]];[...list.children].forEach((b,k)=>b.setAttribute('aria-selected',k===i));
     $('exDetail').innerHTML=`<span class="tag t-${e[1]}" style="border-color:${t[1]}">${t[0]}</span><div class="big t-${e[1]}">${e[2]}</div><div class="muted">${e[3]}</div><h3>The question</h3><p>${e[4]}</p><h3>What I found</h3><p>${e[5]}</p>`}
    pick(6);
    
    // ---------- leakage bars ----------
    $('leak').innerHTML=[['No rows excluded',92.2,'var(--coral)'],['±10 rows',83.2,'var(--purple)'],['±20 rows',60.4,'var(--purple)'],['±50 rows',51.1,'var(--cyan)']].map(r=>`<div class="baritem"><div class="barlabel"><span>${r[0]}</span><b>${r[1]}%</b></div><div class="track"><div class="fill" data-w="${r[1]}" style="background:${r[2]}"></div></div></div>`).join('')+'<p class="fineprint">Nearest-neighbor accuracy across 1,226 walking windows as nearby windows are excluded.</p>';
    const bio=new IntersectionObserver((es,o)=>es.forEach(e=>{if(e.isIntersecting){e.target.querySelectorAll('[data-w]').forEach(f=>f.style.width=f.dataset.w+'%');o.unobserve(e.target)}}),{threshold:.3});document.querySelectorAll('.barsec').forEach(x=>bio.observe(x));
    
    // ---------- effects ----------
    document.querySelectorAll('.metric strong[data-n]').forEach(el=>{const to=+el.dataset.n,d=+el.dataset.d||0,t0=performance.now();const f=t=>{const p=reduce?1:Math.min(1,(t-t0)/1400),e=1-Math.pow(1-p,3);el.textContent=(to*e).toLocaleString(undefined,{minimumFractionDigits:d,maximumFractionDigits:d});if(p<1)requestAnimationFrame(f)};requestAnimationFrame(f)});
    if(!reduce){const rv=new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting){e.target.classList.add('in');rv.unobserve(e.target)}}),{threshold:.12});document.querySelectorAll('.section,.step').forEach(x=>{x.classList.add('reveal');rv.observe(x)});
     addEventListener('pointermove',e=>{document.body.style.setProperty('--mx',e.clientX+'px');document.body.style.setProperty('--my',e.clientY+'px')});
     document.querySelectorAll('.tilt').forEach(x=>{x.addEventListener('pointermove',e=>{const b=x.getBoundingClientRect(),px=(e.clientX-b.left)/b.width-.5,py=(e.clientY-b.top)/b.height-.5;x.style.transform=`perspective(900px) rotateY(${px*6}deg) rotateX(${-py*6}deg)`});x.addEventListener('pointerleave',()=>x.style.transform='')})}
    if(!D){$('walkStats').textContent='Walking viewer needs pca_data.json. Run python export_pca.py first.';return}
    
    // ---------- 3D scene ----------
    const n=D.base.length,ids=D.participants,M=D.modified,uniq=[...new Set(ids)].sort((a,b)=>a-b);
    const sel=$('who');sel.innerHTML='<option value="all">All participants</option>'+uniq.map(u=>`<option value="${u}">Participant ${u}</option>`).join('');
    sel.onchange=()=>{who=sel.value;paintCols()};
    const R=new THREE.WebGLRenderer({canvas:$('scene'),alpha:true,antialias:true});R.setPixelRatio(Math.min(devicePixelRatio,2));
    const scene=new THREE.Scene(),cam=new THREE.PerspectiveCamera(55,1,.1,150),grp=new THREE.Group();cam.position.z=15;scene.add(grp);
    const c64=document.createElement('canvas');c64.width=c64.height=64;const cx=c64.getContext('2d'),gr=cx.createRadialGradient(32,32,0,32,32,32);gr.addColorStop(0,'rgba(255,255,255,1)');gr.addColorStop(.4,'rgba(255,255,255,.5)');gr.addColorStop(1,'rgba(255,255,255,0)');cx.fillStyle=gr;cx.fillRect(0,0,64,64);const spr=new THREE.CanvasTexture(c64);
    const hue=new Map(uniq.map((u,i)=>[u,new THREE.Color().setHSL(.48+.34*i/uniq.length,.85,.7)]));
    const pb=new Float32Array(n*3),pm=new Float32Array(n*3),cb2=new Float32Array(n*3),cm=new Float32Array(n*3);
    for(let i=0;i<n;i++)for(let k=0;k<3;k++)pb[i*3+k]=pm[i*3+k]=D.base[i][k];
    const g1=new THREE.BufferGeometry(),g2=new THREE.BufferGeometry();
    g1.setAttribute('position',new THREE.BufferAttribute(pb,3));g1.setAttribute('color',new THREE.BufferAttribute(cb2,3));
    g2.setAttribute('position',new THREE.BufferAttribute(pm,3));g2.setAttribute('color',new THREE.BufferAttribute(cm,3));
    const mat=s=>new THREE.PointsMaterial({size:s,map:spr,vertexColors:true,transparent:true,depthWrite:false,blending:THREE.AdditiveBlending});
    const m2=mat(.24);grp.add(new THREE.Points(g1,mat(.2)),new THREE.Points(g2,m2));
    const coral=new THREE.Color('#ff7b87');
    paintCols=()=>{for(let i=0;i<n;i++){const d=(who==='all'||+who===ids[i])?1:.1,c=hue.get(ids[i]);
     cb2[i*3]=c.r*d;cb2[i*3+1]=c.g*d;cb2[i*3+2]=c.b*d;cm[i*3]=coral.r*d;cm[i*3+1]=coral.g*d;cm[i*3+2]=coral.b*d}
     g1.attributes.color.needsUpdate=g2.attributes.color.needsUpdate=true};paintCols();
    const mark=new THREE.Sprite(new THREE.SpriteMaterial({map:spr,color:0xffffff,blending:THREE.AdditiveBlending,depthTest:false,transparent:true}));mark.scale.set(1.4,1.4,1);grp.add(mark);
    moveMarker=i=>mark.position.set(D.base[i][0],D.base[i][1],D.base[i][2]);
    let mx=0,my=0;addEventListener('pointermove',e=>{mx=e.clientX/innerWidth-.5;my=e.clientY/innerHeight-.5});
    function size(){R.setSize(innerWidth,innerHeight,false);cam.aspect=innerWidth/innerHeight;grp.position.x=innerWidth>960?3.4:0;cam.updateProjectionMatrix()}
    addEventListener('resize',size);size();
    
    // ---------- walking viewer ----------
    const T=D.traces||[],tc=$('trace'),g=tc.getContext('2d'),LC=['#78e6ff','#b6a0ff','#70f3c1'],LN=['x','y','z'];
    if(!T.length){$('walkStats').textContent='Re-run python export_pca.py to export real walking traces.'}
    else{walker=T[0];const wb=$('walkers');
     T.forEach(t=>{const b=document.createElement('button');b.textContent='Participant '+t.participant;b.onclick=()=>{walker=t;moveMarker(t.index);[...wb.children].forEach(x=>x.setAttribute('aria-pressed',x===b))};b.setAttribute('aria-pressed',t===walker);wb.appendChild(b)});moveMarker(walker.index)}
    function drawTrace(ts){const W=tc.width,H=tc.height;g.clearRect(0,0,W,H);if(!walker)return;
     const sig=walker.signal,N=sig[0].length,lane=H/3,px=60,pw=W-px-20,f=1+gain/100;
     let m=0;for(let c=0;c<3;c++)for(const v of sig[c])m=Math.max(m,Math.abs(v));m*=1.6;
     const ph=reduce?1:(ts/2560)%1,cur=Math.floor(ph*N),mag=[],mag0=[];
     for(let i=0;i<N;i++){const s=sig[0][i]**2+sig[1][i]**2+sig[2][i]**2;mag0.push(Math.sqrt(s));mag.push(Math.sqrt(s)*f)}
     for(let c=0;c<3;c++){const cy=lane*c+lane/2;g.strokeStyle='rgba(170,190,240,.12)';g.beginPath();g.moveTo(px,cy);g.lineTo(W-20,cy);g.stroke();
      g.fillStyle='#9ba9c6';g.font='14px DM Sans, sans-serif';g.fillText('acc '+LN[c],8,cy+4);
      g.lineWidth=1.5;g.strokeStyle='rgba(255,255,255,.28)';g.beginPath();sig[c].forEach((v,i)=>{const x=px+i/(N-1)*pw,y=cy-v/m*lane/2;i?g.lineTo(x,y):g.moveTo(x,y)});g.stroke();
      g.lineWidth=2.5;g.strokeStyle=LC[c];g.shadowColor=LC[c];g.shadowBlur=14;g.beginPath();for(let i=0;i<=cur;i++){const x=px+i/(N-1)*pw,y=cy-sig[c][i]*f/m*lane/2;i?g.lineTo(x,y):g.moveTo(x,y)}g.stroke();g.fillStyle=LC[c];g.beginPath();g.arc(px+cur/(N-1)*pw,cy-sig[c][cur]*f/m*lane/2,5,0,6.283);g.fill();g.shadowBlur=0}
     const mean=mag.reduce((a,b)=>a+b)/N,sd=Math.sqrt(mag.reduce((a,b)=>a+(b-mean)**2,0)/N),pk=[];
     for(let i=2;i<N-2;i++)if(mag[i]>mean+.6*sd&&mag[i]>=mag[i-1]&&mag[i]>=mag[i+1]&&(!pk.length||i-pk[pk.length-1]>=12))pk.push(i);
     pk.forEach(i=>{if(i<=cur){g.fillStyle='#ff7b87';g.beginPath();g.arc(px+i/(N-1)*pw,10,5,0,6.283);g.fill()}});
     const cxp=px+ph*pw;g.strokeStyle='rgba(255,255,255,.45)';g.lineWidth=1;g.beginPath();g.moveTo(cxp,0);g.lineTo(cxp,H);g.stroke();
     $('walkStats').textContent=`${pk.length} acceleration peaks in 2.56 s (about ${Math.round(pk.length*60/2.56)} per minute). Largest peak ${Math.max(...mag).toFixed(2)} g vs ${Math.max(...mag0).toFixed(2)} g original.`}
    
    // ---------- render loop ----------
    function frame(ts){const fl=gain/5,lo=Math.floor(fl),hi=Math.min(10,lo+1),t=fl-lo,a=M[lo],b=M[hi];
     for(let i=0;i<n;i++)for(let k=0;k<3;k++){const j=i*3+k;pm[j]+=(mix(a[i][k],b[i][k],t)-pm[j])*(reduce?1:.1)}
     g2.attributes.position.needsUpdate=true;burst*=.93;m2.size=.24+burst*.35;
     const s=1.3+(reduce?0:.25*Math.sin(ts/300));mark.scale.set(s,s,1);
     if(!reduce)grp.rotation.y+=.0006+mx*.002;
     grp.rotation.x+=(my*.5-grp.rotation.x)*.04;
     const sp=Math.min(1,scrollY/Math.max(1,document.body.scrollHeight-innerHeight));cam.position.z+=(15-4*Math.sin(sp*Math.PI)-cam.position.z)*.05;grp.rotation.z+=(sp*.9-grp.rotation.z)*.04;
     R.render(scene,cam);drawTrace(ts);requestAnimationFrame(frame)}
    requestAnimationFrame(frame);
    })();