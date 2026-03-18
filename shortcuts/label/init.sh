[ $# -ge 1 ] && tag=$1
[ $# -ge 1 ] || tag=select

echo $tag > /tmp/tag

# script.google.com
# function etiquetarCorreosPorSubject() {
#   const subjects = [
# "subject",
#   ];
#
#   const labelName = "label";
#   let label = GmailApp.getUserLabelByName(labelName);
#   if (!label) {
#     label = GmailApp.createLabel(labelName);
#   }
#   subjects.forEach(subject => {
#     const threads = GmailApp.search(`subject:"${subject}"`);
#     threads.forEach(thread => {
#       thread.addLabel(label);
#     });
#
#     Logger.log(`Procesado subject: ${subject} | Encontrados: ${threads.length}`);
#   });
# }
