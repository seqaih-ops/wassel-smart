(function(root){
function predict(model, x) {
 const d=Number(x.Distance_Km), o=Number(x.Driver_Orders), t=Number(x.Traffic_Level);
 if(!Number.isFinite(d)||d<1||d>50||!Number.isInteger(o)||o<0||o>10||![1,2,3].includes(t)) throw Error('أدخل مسافة من 1 إلى 50 كم، وعدد طلبات صحيحا من 0 إلى 10، ومستوى ازدحام من 1 إلى 3.');
 const raw=[d,o,t,d*t,o*t];
 const z=raw.map((v,i)=>Math.fround((v-model.mean[i])/model.scale[i]));
 let score=model.initial_log_odds;
 for(const tree of model.trees){let k=0;while(tree.left[k]!==-1)k=z[tree.feature[k]]<=tree.threshold[k]?tree.left[k]:tree.right[k];score+=model.learning_rate*tree.value[k];}
 return 1/(1+Math.exp(-score));
}
if(typeof module!=='undefined')module.exports={predict}; else root.Wassel={predict};
})(globalThis);
