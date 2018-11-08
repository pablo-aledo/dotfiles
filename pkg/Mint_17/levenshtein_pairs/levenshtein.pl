#!/usr/bin/perl -w
use strict;
use Text::Levenshtein qw(distance);
use Getopt::Std;

our $opt_n;
getopts('n:');
$opt_n ||= -1; # print all the matches if -n is not provided

my @lines=<>;
my %distances = ();

# for each combination of two lines, compute distance
foreach(my $i=0; $i <= $#lines - 1; $i++) {
  foreach(my $j=$i + 1; $j <= $#lines; $j++) {
        my $d = distance($lines[$i], $lines[$j]);
        push @{ $distances{$d} }, $lines[$i] . $lines[$j];
  }
}

# print in order of increasing distance
foreach my $d (sort { $a <=> $b } keys %distances) {
  print "At distance $d:\n" . join("\n", @{ $distances{$d} }) . "\n";
  last unless --$opt_n;
}
