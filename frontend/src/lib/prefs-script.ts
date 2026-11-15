// Plain (server-importable) module: the pre-paint preferences script used by app/layout.tsx.
export const PREFS_KEY = "zalu.prefs";

/**
 * Runs in <head> before the body paints. Kept dependency-free and tolerant of bad storage.
 * Sets data-theme (resolved light/dark), data-theme-pref, data-scale, data-motion (effective level, 0 when the OS asks
 * for reduced motion) and color-scheme for native controls.
 */
export const PREFS_SCRIPT = `(function(){try{var d=document.documentElement,p={};try{p=JSON.parse(localStorage.getItem("${PREFS_KEY}")||"{}")||{}}catch(e){}
var t=p.theme==="light"||p.theme==="dark"?p.theme:"system",s=p.scale==="compact"||p.scale==="large"?p.scale:"default",m=[0,1,2,3].indexOf(p.motion)>=0?p.motion:2;
var dark=t==="dark"||(t==="system"&&matchMedia("(prefers-color-scheme: dark)").matches);
if(matchMedia("(prefers-reduced-motion: reduce)").matches)m=0;
d.dataset.theme=dark?"dark":"light";d.dataset.themePref=t;d.dataset.scale=s;d.dataset.motion=String(m);d.style.colorScheme=dark?"dark":"light";}catch(e){}})();`;
