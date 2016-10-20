python := python2

data-dir := data

ceesr-data := \
	g1ib.flac \
	g1ip.flac \
	g1it.flac \
	g1jb.flac \
	g1jp.flac \
	g1jt.flac \
	g2ib.flac \
	g2ip.flac \
	g2it.flac \
	g2jb.flac \
	g2jp.flac \
	g2jt.flac \
	g5ib.flac \
	g5ip.flac \
	g5it.flac \
	g5jb.flac \
	g5jp.flac \
	g5jt.flac \
	gi.flac \
	gi.txt \
	gj.flac \
	gj.txt \
	s1ib.flac \
	s1ip.flac \
	s1it.flac \
	s1jb.flac \
	s1jp.flac \
	s1jt.flac \
	s2ib.flac \
	s2ip.flac \
	s2it.flac \
	s2jb.flac \
	s2jp.flac \
	s2jt.flac \
	s5ib.flac \
	s5ip.flac \
	s5it.flac \
	s5jb.flac \
	s5jp.flac \
	s5jt.flac \
	si.flac \
	si.txt \
	sj.flac \
	sj.txt

ceesr-data-merged := g.txt s.txt

ceesr-data-url := http://universe.hiit.fi/data/ceesr/

ceesr-data-files := $(addprefix $(data-dir)/,$(ceesr-data))
ceesr-data-merged-files := $(addprefix $(data-dir)/,$(ceesr-data-merged))

noisex-92-samplerate := 19980

noisex-92-data := \
	white.mat \
	pink.mat \
	babble.mat \
	factory1.mat \
	factory2.mat \
	buccaneer1.mat \
	buccaneer2.mat \
	f16.mat \
	destroyerengine.mat \
	destroyerops.mat \
	leopard.mat \
	m109.mat \
	machinegun.mat \
	volvo.mat \
	hfchannel.mat

noisex-92-data-url := http://spib.linse.ufsc.br/data/noise/

noisex-92-data-files := $(addprefix $(data-dir)/,$(noisex-92-data))
noisex-92-raw-files := $(addsuffix .raw,$(basename $(noisex-92-data-files)))
noisex-92-flac-files := $(addsuffix .flac,$(basename $(noisex-92-data-files)))

.PHONY: all
all: $(ceesr-data-files) $(ceesr-data-merged-files) $(noisex-92-flac-files)

$(ceesr-data-files): $(data-dir)/%: | $(data-dir)
	curl -u ceesr:ceesr -o $@ $(ceesr-data-url)$(notdir $@)

$(ceesr-data-merged-files): %.txt: %i.txt %j.txt
	$(python) mergelabels.py $^ > $@

$(noisex-92-data-files): $(data-dir)/%: | $(data-dir)
	curl -o $@ $(noisex-92-data-url)$(notdir $@)

$(noisex-92-raw-files): %.raw: %.mat
	$(python) mat2raw.py $< $@

$(noisex-92-flac-files): %.flac: %.raw
	ffmpeg -y -f s16le -ar $(noisex-92-samplerate) -ac 1 -i $< $@

.PHONY: clean-noisex-92-flac
clean-noisex-92-flac:
	rm -f $(noisex-92-flac-files)

.PHONY: clean-noisex-92-raw
clean-noisex-92-raw:
	rm -f $(noisex-92-raw-files)

.PHONY: clean-noisex-92-data
clean-noisex-92-data:
	rm -f $(noisex-92-data-files)

.PHONY: clean-noisex-92
clean-noisex-92: clean-noisex-92-flac clean-noisex-92-raw clean-noisex-92-data

$(data-dir):
	mkdir $@
