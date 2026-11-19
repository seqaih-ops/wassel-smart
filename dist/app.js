'use strict';
const $=id=>document.getElementById(id);
let model,metrics,currentRows=[];
const trafficText=['','منخفض','متوسط','مرتفع'];
const pct=p=>(p*100).toFixed(1)+'%';
function values(){const ids=['distance','orders','traffic','alt-orders','alt-distance','alt-traffic'];if(ids.some(id=>$(id).value.trim()===''))throw Error('أكمل جميع المدخلات أولا.');return ids.map(id=>Number($(id).value));}
function compare(){
 try{
  const [d,o,t,ao,ad,at]=values();
  const specs=[['الوضع الحالي',d,o,t],['سائق بديل',d,ao,t],['مسار بديل',ad,o,at]];
  const rows=specs.map(([name,d,o,t])=>({name,Distance_Km:d,Driver_Orders:o,Traffic_Level:t,p:Wassel.predict(model,{Distance_Km:d,Driver_Orders:o,Traffic_Level:t})}));
  currentRows=rows;$('error').textContent='';
  const p=rows[0].p;$('risk').textContent=pct(p);$('current-summary').textContent=`${d} كم · ${o} طلبات · ازدحام ${trafficText[t]}`;
  $('risk-label').textContent=`مستوى ${p<.3?'منخفض':p<.6?'متوسط':'مرتفع'} · ${p>=.5?'متأخر متوقع':'غير متأخر متوقع'}`;
  const lowest=Math.min(...rows.map(r=>r.p));const best=rows.filter(r=>Math.abs(r.p-lowest)<1e-9);
  let message;
  if(best.length===3)message='تعادلت الخيارات الثلاثة في تقدير النموذج. لا يظهر تحسن متوقع من التغييرات المدخلة.';
  else if(best.some(r=>r.name===rows[0].name))message='الوضع الحالي ضمن الخيارات الأقل احتمالا للتأخر؛ لا يظهر بديل أفضل في المقارنة الحالية.';
  else message=`الأقل احتمالا بين الخيارات المدخلة: ${best.map(r=>r.name).join(' و')}، بانخفاض مقداره ${((p-lowest)*100).toFixed(1)} نقطة مئوية. تحقق من قابلية تنفيذ الخيار قبل اتخاذ القرار.`;
  $('recommendation').textContent=message;
  $('scenarios').innerHTML=rows.map(r=>{let delta=(p-r.p)*100;return `<article class="scenario"><div class="scenario-top"><div><strong>${r.name}</strong><small>${r.Distance_Km} كم · ${r.Driver_Orders} طلبات · ازدحام ${trafficText[r.Traffic_Level]}</small></div><span class="percent">${pct(r.p)}</span></div><div class="bar"><i style="width:${r.p*100}%"></i></div><span class="delta">${r===rows[0]?'مرجع المقارنة':Math.abs(delta)<.05?'دون فرق ملحوظ':delta>0?`انخفاض ${delta.toFixed(1)} نقطة مئوية`:`ارتفاع ${(-delta).toFixed(1)} نقطة مئوية`}</span></article>`}).join('');
  $('download').disabled=false;
 }catch(e){$('error').textContent=e.message;$('download').disabled=true;}
}
$('form').addEventListener('submit',e=>{e.preventDefault();compare()});
$('form').addEventListener('input',()=>{$('download').disabled=true;$('recommendation').textContent='تغيّرت المدخلات. اضغط «قارن السيناريوهات» لتحديث النتائج المعروضة.'});
$('reset').addEventListener('click',()=>{HTMLFormElement.prototype.reset.call($('form'));compare()});
$('download').addEventListener('click',()=>{const lines=['Scenario,Distance_Km,Driver_Orders,Traffic_Level,Delay_Probability,Predicted_Delayed,Data_Source',...currentRows.map(r=>[r.name,r.Distance_Km,r.Driver_Orders,r.Traffic_Level,r.p.toFixed(6),r.p>=.5?1:0,'synthetic_educational'].join(','))];const url=URL.createObjectURL(new Blob(['\ufeff'+lines.join('\n')],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download='wassel_scenarios.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)});
Promise.all([fetch('model.json').then(r=>{if(!r.ok)throw Error('model');return r.json()}),fetch('metrics.json').then(r=>{if(!r.ok)throw Error('metrics');return r.json()})]).then(([m,e])=>{
 model=m;metrics=e;const tuned=e.models.find(r=>r.model==='Tuned Gradient Boosting');
 $('metrics').innerHTML=[[tuned.AP.toFixed(3),'متوسط الدقة AP'],[tuned.F1.toFixed(3),'F1 عند حد 0.5'],[e.train_records.toLocaleString('en'),'سجل تدريب اصطناعي'],[e.test_records.toLocaleString('en'),'سجل اختبار اصطناعي']].map(([v,l])=>`<div class="metric"><strong>${v}</strong><span>${l}</span></div>`).join('');
 $('model-table').innerHTML=e.models.map(r=>`<tr><td>${r.model}</td><td>${r.AP.toFixed(3)}</td><td>${r.ROC_AUC.toFixed(3)}</td><td>${r.F1.toFixed(3)}</td></tr>`).join('');
 const labels=['المسافة','عدد الطلبات','الازدحام','المسافة × الازدحام','الطلبات × الازدحام'];
 $('importance').innerHTML=m.importance.map((v,i)=>({v,l:labels[i]})).sort((a,b)=>b.v-a.v).map(r=>`<div class="imp">${r.l}<span>${pct(r.v)}</span><div class="bar"><i style="width:${r.v*100}%"></i></div></div>`).join('');
 $('compare').disabled=false;$('compare').textContent='قارن السيناريوهات';compare();
}).catch(()=>{$('error').textContent='تعذر تحميل النموذج. أعد تحميل الصفحة أو شغّل النسخة المحلية بخادم HTTP وفق دليل المشروع.';$('compare').textContent='تعذر تحميل النموذج';$('recommendation').textContent='النتائج غير متاحة حتى يتم تحميل النموذج.'});
